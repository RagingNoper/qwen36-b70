# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEQuantConfig,
    )

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.logger import init_logger

logger = init_logger(__name__)


class Int8OnlineMoEMethod(OnlineMoEMethodBase):
    """Online per-channel INT8 MoE quantization.
    Loads fp16/bf16 weights and quantizes them per-row to int8 during loading.
    """

    def __init__(
        self,
        *,
        layer: torch.nn.Module,
    ):
        super().__init__(layer.moe_config)
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_weights(layer)
        self._setup_kernel(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(self, layer: Module) -> None:
        vmax = torch.iinfo(torch.int8).max

        w13 = torch.empty_like(layer.w13_weight, dtype=torch.int8)
        w2 = torch.empty_like(layer.w2_weight, dtype=torch.int8)
        w13_scale = torch.zeros(
            layer.num_experts,
            layer.w13_weight.shape[1],
            device=w13.device,
            dtype=torch.float32,
        )
        w2_scale = torch.zeros(
            layer.num_experts,
            layer.w2_weight.shape[1],
            device=w2.device,
            dtype=torch.float32,
        )

        for expert in range(layer.local_num_experts):
            # w13: per-row quantization over hidden_size dim
            w = layer.w13_weight[expert, :, :]
            scales = w.abs().amax(dim=1) / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w13[expert, :, :] = q.to(torch.int8)
            w13_scale[expert, :] = scales

            # w2: per-row quantization over intermediate_size dim
            w = layer.w2_weight[expert, :, :]
            scales = w.abs().amax(dim=1) / vmax
            q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax)
            w2[expert, :, :] = q.to(torch.int8)
            w2_scale[expert, :] = scales

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_scale", w13_scale)
        replace_parameter(layer, "w2_scale", w2_scale)

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_int8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        # [xpu-graph-work] batch-1 decode fast path: pre-allocate persistent
        # buffers (stable addr for graph capture) for the custom int8 GEMV that
        # replaces the occupancy-starved grouped GEMM at small T.
        import os as _os
        self._gemv_max_t = int(_os.environ.get("VLLM_INT8_GEMV_MAX_T", "2"))
        self._gemv_warned = False
        if self._gemv_max_t > 0:
            twoI = layer.w13_weight.shape[1]
            H = layer.w2_weight.shape[1]
            topk = getattr(layer, "top_k", None) or self.moe.experts_per_token
            dev = layer.w13_weight.device
            self._gemv_topk = topk
            # [xpu-graph-work] PER-T buffers: each cudagraph capture size (T) gets
            # its OWN h/out so the size-1 and size-2 graphs never alias the same
            # storage (that aliasing was the T>=2 NaN/corruption bug -> UPDATE 18).
            self._gemv_h = {t: torch.empty((t * topk, twoI // 2), dtype=torch.bfloat16, device=dev)
                            for t in range(1, self._gemv_max_t + 1)}
            self._gemv_out = {t: torch.empty((t, H), dtype=torch.bfloat16, device=dev)
                              for t in range(1, self._gemv_max_t + 1)}

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> "FusedMoEQuantConfig | None":
        # [xpu-graph-work] The stock make_int8_moe_quant_config downgrades to
        # W8A16 (dequant int8 weight -> bf16 matmul, NO integer DPAS) whenever
        # activation scales are absent. But the backend was selected for
        # kInt8DynamicTokenSym (dynamic per-token int8 activations). Build the
        # real W8A8 config (a1/a2 dynamic, per_act_token_quant=True) so the
        # kernel does int8xint8 -> DPAS on Xe2 instead of dequant-to-bf16.
        from vllm.model_executor.layers.fused_moe.config import (
            int8_w8a8_moe_quant_config,
        )
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            a1_scale=None,
            a2_scale=None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_act_token_quant=True,
        )

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    ):
        # [xpu-graph-work] batch-1 decode: route small T to the custom int8 GEMV
        # (direct expert-indexed streaming, beats the occupancy-starved grouped
        # GEMM at ~1 row/expert). Falls back to the grouped kernel otherwise.
        T = x.shape[0]
        # layer.activation is the MoEActivation enum (name "SILU"), not a string.
        act_is_silu = getattr(layer.activation, "name", str(layer.activation)) == "SILU"
        shared_ok = shared_experts is None or (
            shared_experts_input is not None
            and getattr(shared_experts, "_layer", None) is not None
        )
        if (getattr(self, "_gemv_max_t", 0) > 0
                and T <= self._gemv_max_t
                and shared_ok
                and getattr(layer, "expert_map", None) is None
                and not getattr(layer, "apply_router_weight_on_input", False)
                and act_is_silu
                and topk_ids.shape[-1] == self._gemv_topk):
            from ._int8_gemv import moe_int8_gemv
            if not self._gemv_warned:
                logger.info_once("[xpu-graph-work] int8 GEMV decode fast path ACTIVE (T<=%d)",
                                 self._gemv_max_t)
                self._gemv_warned = True
            # apply() returns ROUTED-ONLY (runner adds shared). Per-T buffers.
            return moe_int8_gemv(
                x, layer.w13_weight, layer.w13_scale,
                layer.w2_weight, layer.w2_scale,
                topk_weights, topk_ids,
                self._gemv_h[T], self._gemv_out[T],
            )
        return super().apply(
            layer, x, topk_weights, topk_ids,
            shared_experts, shared_experts_input,
        )
