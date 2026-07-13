# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.int8 import (
    Int8OnlineMoEMethod,
)


class ExpertsInt8Config(QuantizationConfig):
    """Online int8 quantization for MoE expert weights.
    Linear layers are left unquantized.

    Backward-compatible config for ``--quantization experts_int8``.
    Prefer ``--quantization int8_per_channel``
    """

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "experts_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExpertsInt8Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        import os
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        # [xpu-graph-work] Plan1/2.1: int8 lm_head (dense W8A8 GEMV decode fast
        # path). Env-gated; untied lm_head only (embed stays bf16).
        if (isinstance(layer, ParallelLMHead)
                and os.environ.get("VLLM_INT8_LMHEAD", "0") == "1"):
            from vllm.model_executor.layers.quantization.online._int8_linear import (
                get_int8_lmhead_method,
            )
            return get_int8_lmhead_method()
        if isinstance(layer, LinearBase):
            # [xpu-graph-work] Plan1/2.2: int8 the big attention/GDN projections.
            if os.environ.get("VLLM_INT8_ATTN", "0") == "1":
                from vllm.model_executor.layers.quantization.online._int8_linear import (
                    want_int8_linear, get_int8_linear_method,
                )
                if want_int8_linear(prefix):
                    return get_int8_linear_method()
            return UnquantizedLinearMethod()
        elif isinstance(layer, RoutedExperts):
            return Int8OnlineMoEMethod(layer=layer)
        return None
