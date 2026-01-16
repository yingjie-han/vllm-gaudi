import math
import os
from functools import partial
from typing import Optional, Callable, Union
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm.logger import init_logger
from vllm.distributed import parallel_state

from transformers import BatchFeature
from vllm.transformers_utils.processor import (cached_image_processor_from_config)
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig

from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionAttention, Qwen2VisionBlock, Qwen2VisionTransformer, Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration, Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo, Qwen2VLVideoInputs,
    Qwen2VLImageInputs, Qwen2VLImageEmbeddingInputs, Qwen2VLImagePixelInputs, Qwen2VLVideoEmbeddingInputs,
    Qwen2VLVideoPixelInputs, Qwen2VLProcessor)

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm.config import MultiModalConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.utils import maybe_prefix

from vllm.multimodal.inputs import MultiModalFieldConfig

import habana_frameworks.torch as htorch
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import FusedSDPA

logger = init_logger(__name__)

class AttentionLongSequence:

    @staticmethod
    def forward(q, k, v, mask, q_block_size, softmax_mode):
        """
        Support long sequence at prompt phase
        """
        q_len = q.size(-2)
        assert q_len % q_block_size == 0
        q_tiles = (q_len //
                   q_block_size) if (q_len % q_block_size == 0) else math.ceil(
                       q_len / q_block_size)
        attn_output = torch.zeros_like(q)

        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = q[:, :, s:e, :]
            row_mask = mask[:, :, s:e, :]
            attn_output[:, :,
                        s:e, :] = FusedSDPA.apply(row_q, k, v, row_mask, 0.0,
                                                  False, None, softmax_mode)
            # TODO: markstep after a couple of iterations
            # need to experiment the optimal number.
            if i % 75 == 0:
                htcore.mark_step()
        return attn_output


def create_block_diagonal_attention_mask_outerprod(indices):
    maxsize = indices[-1]
    range_to_max_for_each_img = torch.arange(
        maxsize,
        device=indices.device).unsqueeze(0).repeat(indices.shape[0] - 1, 1)
    lesser = range_to_max_for_each_img < indices[1:].unsqueeze(1)
    greater_eq = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
    range_indices = torch.logical_and(lesser, greater_eq).float()
    # can reduce sum externally or as batchmatmul
    if range_indices.shape[-1] > 40000:
        log_msg = "einsum running on CPU :" + str(range_indices.shape)
        logger.info(log_msg)
        range_indices = range_indices.to("cpu")
        res = torch.einsum('bi,bj->ij', range_indices, range_indices)
        res = res.to("hpu")
    else:
        res = torch.einsum('bi,bj->ij', range_indices, range_indices)
    return res.bool()

class HPUQwen2VisionAttention(Qwen2VisionAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=projection_size,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )

        self.softmax_mode = 'fp32' if os.environ.get('VLLM_FP32_SOFTMAX_VISION', 'false').lower() in ['true', '1'
                                                                                                      ] else 'None'

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, 3 * head * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)

        q, k, v = (rearrange(x, "s b ... -> b s ...") for x in (q, k, v))

        # [2 * b, s, heads, head_dim]
        qk_concat = torch.cat([q, k], dim=0)
        qk_rotated = self.apply_rotary_emb(
            qk_concat,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        )
        q, k = torch.chunk(qk_rotated, 2, dim=0)

        q1, k1, v1 = (rearrange(x, "b s h d -> b h s d")
                        for x in [q, k, v])
        (batch_size, _, seq_len_N_t, _) = q1.shape
        (batch_size, _, seq_len_N_s, _) = k1.shape

        attn_mask = cu_seqlens if cu_seqlens is not None else None

        if q1.shape[2] <= 65536:  # need to investigate this crosspoint
            fused_out = FusedSDPA.apply(q1, k1, v1, attn_mask, 0.0, False,
                                        None, self.softmax_mode)
        else:
            fused_out = AttentionLongSequence.forward(
                q1, k1, v1, attn_mask, 64, self.softmax_mode)

        context_layer = rearrange(fused_out, "b h s d -> s b (h d)").contiguous()


        output, _ = self.proj(context_layer)
        return output

class HPUQwen2VisionBlock(Qwen2VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: type[nn.Module] = QuickGELU,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )
        self.attn = HPUQwen2VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionTransformerStaticShape(Qwen2VisionTransformer):
    """
    Here we overwrite some of the methods of Qwen2VisionTransformer
    to make the model more friendly to static shapes. Specifically,
    we split the forward  method into:
      - pre_attn (dynamic)
      - forward (static shape)
      - post_attn (dynamic)
    and we should call get_image_embeds instead of forward, allowing
    the forward method ro run with HPU_Graphs, whereas the
    pre_attn and post_attn methods are allow to be dynamic.
    """

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.compose_seq_len = 1024

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        embed_dim = vision_config.embed_dim
        depth = vision_config.depth
        num_heads = vision_config.num_heads
        mlp_ratio = vision_config.mlp_ratio

        self.blocks = nn.ModuleList([
            HPUQwen2VisionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                    multimodal_config=multimodal_config,
            ) for layer_idx in range(depth)
        ])

    def pad_multimodal_data(self,
                            pixel_values,
                            image_grid_thw,
                            vision_buckets,
                            constant_value=0):
        assert pixel_values.shape[0] % 4 == 0, 'needs 64 aligned resolution'

        desired_number_of_pixels = vision_buckets.get_multimodal_bucket(
            pixel_values.shape[0])
        padding_len = desired_number_of_pixels - pixel_values.shape[0]
        if padding_len <= 0:
            return pixel_values, image_grid_thw

        logger_msg = "Padding current number pixel " \
            + str(pixel_values.shape[0]) \
            + " to " \
            + str(desired_number_of_pixels)
        logger.debug(logger_msg)

        pixel_values = torch.cat([
            pixel_values,
            torch.ones((padding_len, pixel_values.shape[1]), \
                device=pixel_values.device) * constant_value
        ])

        image_grid_thw = torch.cat([
            image_grid_thw,
            torch.tensor([[1, 2, padding_len // 2]],
                         device=image_grid_thw.device)
        ])

        return pixel_values, image_grid_thw

    def pre_attn(self, x: torch.Tensor, grid_thw: torch.Tensor):
        # patchify
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # compute position embedding
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        return hidden_states, rotary_pos_emb_cos, rotary_pos_emb_sin, cu_seqlens

    def forward(self, x: torch.Tensor, fullattn_mask: Optional[torch.Tensor],
                rotary_pos_emb_cos: torch.Tensor, rotary_pos_emb_sin: torch.Tensor,
                bypass_hpu_graphs =False) -> torch.Tensor:

        hidden_states = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            htcore.mark_step()
            hidden_states = blk(hidden_states,
                                cu_seqlens=fullattn_mask,
                                rotary_pos_emb_cos=rotary_pos_emb_cos,
                                rotary_pos_emb_sin=rotary_pos_emb_sin)
        return hidden_states

    def post_attn(self, hidden_states: torch.Tensor):
        # adapter
        hidden_states = self.merger(hidden_states)

        return hidden_states

    def get_image_embeds(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        vision_buckets,
    ) -> torch.Tensor:

        offset = 0
        results = []
        calc_img_size = 0
        calc_img_len_list = []
        calc_grid_thw_list = []
        # process each image one by one
        for img_idx in range(grid_thw.shape[0]):
            img_shape = grid_thw[img_idx, :].unsqueeze(0)
            curr_img_size = img_shape.prod()
            attn_mask = None
            next_img_size = 100000
            if img_idx < grid_thw.shape[0] - 1:
                img_shape_next = grid_thw[img_idx + 1, :].unsqueeze(0)
                next_img_size = img_shape_next.prod()

            calc_img_size += curr_img_size
            calc_img_len_list.append(curr_img_size)
            calc_grid_thw_list.append(img_shape)
            if calc_img_size + next_img_size < self.compose_seq_len:
                #compose small images SDPA into one bigger SDPA with mask
                continue
            else:
                curr_img_size = calc_img_size
                calc_img_size = 0
                if len(calc_img_len_list) > 1:
                    bucket_img_size = vision_buckets.get_multimodal_bucket(
                        curr_img_size)
                    attn_mask = torch.zeros(bucket_img_size,
                                            bucket_img_size).bool()
                    img_start = 0
                    for img_len in calc_img_len_list:
                        img_end = img_start + img_len
                        attn_mask[img_start:img_end, img_start:img_end] = True
                        img_start = img_end

                    img_shape = torch.cat(calc_grid_thw_list)

                    attn_mask = attn_mask.to(device=self.device)
                calc_img_len_list = []
                calc_grid_thw_list = []

            pixel_values_curr_img = pixel_values[offset:offset +
                                                 curr_img_size, :]

            offset += curr_img_size
            pixel_values_curr_img_padded, img_shape_padded = \
                self.pad_multimodal_data(pixel_values_curr_img, \
                    img_shape, vision_buckets=vision_buckets,constant_value=0)

            pixel_values_curr_img_padded, rot_pos_emb_cos, rot_pos_emb_sin, \
                cu_seqlens = self.pre_attn(
            pixel_values_curr_img_padded, img_shape_padded)

            assert pixel_values_curr_img_padded.shape[0] == \
                 rot_pos_emb_cos.shape[0] == rot_pos_emb_sin.shape[0]

            extra_forward_kwargs = {}
            if htorch.utils.internal.is_lazy():
                padded_len = pixel_values_curr_img_padded.shape[0]
                use_graph = vision_buckets.use_graph(padded_len)
                extra_forward_kwargs.update(
                    {"bypass_hpu_graphs": not use_graph})

            htcore.mark_step()
            hidden_states = self.forward(pixel_values_curr_img_padded,
                                         fullattn_mask=attn_mask,
                                         rotary_pos_emb_cos=rot_pos_emb_cos,
                                         rotary_pos_emb_sin=rot_pos_emb_sin,
                                         **extra_forward_kwargs)
            htcore.mark_step()

            image_embeds = self.post_attn(hidden_states)
            # slice image_embeds to remove the padded parts
            pad_index = curr_img_size// self.spatial_merge_unit
            results += [image_embeds[:pad_index, :]]
        results_cat = torch.concat(results)
        image_embeds = results_cat
        return image_embeds

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=Qwen2VLProcessingInfo,
    dummy_inputs=Qwen2VLDummyInputsBuilder,
)
class HpuQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if hasattr(self, "visual") and self.visual is not None:
            self.visual = Qwen2VisionTransformerStaticShape(
                self.config.vision_config,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                quant_config=vllm_config.quant_config,
                multimodal_config=self.multimodal_config,
                prefix=maybe_prefix(prefix, "visual"),
            )


    def _process_image_input(
        self, image_input: Qwen2VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"]
        else:
            pixel_values = image_input["pixel_values"]

            # if self.use_data_parallel:
            #     return run_dp_sharded_mrope_vision_model(
            #         self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            #     )
            # else:
            #     image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
            image_embeds = self.visual.get_image_embeds(
                pixel_values,
                grid_thw=grid_thw,
                vision_buckets=self.vision_bucket_manager,
            )

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)


    def _process_video_input(
        self, video_input: Qwen2VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"]
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            # if self.use_data_parallel:
            #     return run_dp_sharded_mrope_vision_model(
            #         self.visual,
            #         pixel_values_videos,
            #         grid_thw.tolist(),
            #         rope_type="rope_3d",
            #     )
            # else:
            #     video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
            video_embeds = self.visual.get_image_embeds(
                pixel_values_videos,
                grid_thw=grid_thw,
                vision_buckets=self.vision_buckets,
            )

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)