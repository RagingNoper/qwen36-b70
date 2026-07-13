"""int8 W8A8 dense linear (batch-1 decode fast path) for lm_head / attention
projections on XPU. Per-out-channel int8 weight + fused per-token int8 activation
quant, int8xint8->int32 DPAS-path MAC, dequant. Routes only small T to the int8
GEMV (re-reads weight per row -> optimal at T=1, our single-stream target);
high-T falls back to the stock bf16 method (kept weight) to avoid regressing
concurrency. Memory-freeing (drop bf16) is a follow-up once a batched int8 GEMM
handles high T.
"""
import os
import torch
import triton
import triton.language as tl


def _np2(x):
    return 1 << (x - 1).bit_length()


@triton.jit
def _linear_i8(x, w, ws, y, K: tl.constexpr, N,
               BN: tl.constexpr, BK: tl.constexpr, BLK: tl.constexpr):
    t = tl.program_id(0)
    nb = tl.program_id(1)
    on = nb * BN + tl.arange(0, BN)
    nmask = on < N
    full = tl.arange(0, BLK); km = full < K
    xr = tl.load(x + t * K + full, mask=km, other=0.0).to(tl.float32)
    xs = tl.maximum(tl.max(tl.abs(xr), 0) / 127.0, 1e-8); inv = 1.0 / xs
    acc = tl.zeros((BN,), tl.int32)
    wb = w + on[:, None] * K
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        xf = tl.load(x + t * K + ok).to(tl.float32)
        xq = tl.minimum(tl.maximum(tl.floor(xf * inv + 0.5), -127.0), 127.0).to(tl.int32)
        acc += tl.sum(tl.load(wb + ok[None, :], mask=nmask[:, None], other=0).to(tl.int32) * xq[None, :], 1)
    sw = tl.load(ws + on, mask=nmask, other=0.0)
    tl.store(y + t * N + on, (acc.to(tl.float32) * xs * sw).to(tl.bfloat16), mask=nmask)


# per-in-dim (BN, BK, W). lm_head K=2048 -> tuned by dense_i8 sweep-ish defaults.
_CFG = {2048: (64, 256, 8)}


def int8_linear_forward(x, w_i8, w_s):
    T, K = x.shape
    N = w_i8.shape[0]
    x = x.contiguous()
    y = torch.empty((T, N), dtype=torch.bfloat16, device=x.device)
    bn, bk, w = _CFG.get(K, (64, 256, 8))
    _linear_i8[(T, triton.cdiv(N, bn))](x, w_i8, w_s, y,
        K=K, N=N, BN=bn, BK=bk, BLK=_np2(K), num_warps=w)
    return y


def quant_weight_per_outchannel(w):
    """[N,K] bf16 -> int8 [N,K] + fp32 scale [N] (symmetric per-out-channel)."""
    s = (w.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
    return q, s.float().contiguous()


def _make_lmhead_method():
    # imported lazily so the module is importable without vLLM present (self-test)
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    class Int8LMHeadMethod(UnquantizedEmbeddingMethod):
        """int8 lm_head: quantize weight at load; int8 GEMV for small T, bf16 else."""

        def __init__(self):
            super().__init__()
            self._max_t = int(os.environ.get("VLLM_INT8_LINEAR_MAX_T", "1"))

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            qi, s = quant_weight_per_outchannel(w)
            layer.weight_int8 = qi
            layer.weight_scale = s

        def apply(self, layer, x, bias=None):
            if os.environ.get("VLLM_INT8_DIAG", "0") == "1":
                n = getattr(self, "_dn", 0)
                if n < 20:
                    self._dn = n + 1
                    import sys
                    print(f"[XGW lmhead] T={x.shape[0]} dim={x.dim()} "
                          f"path={'int8' if x.shape[0] <= self._max_t else 'bf16'}",
                          file=sys.stderr, flush=True)
            if x.shape[0] <= self._max_t and hasattr(layer, "weight_int8"):
                y = int8_linear_forward(x, layer.weight_int8, layer.weight_scale)
                return y if bias is None else y + bias
            return super().apply(layer, x, bias)

    return Int8LMHeadMethod


def get_int8_lmhead_method():
    return _make_lmhead_method()()


# --- attention / dense LinearBase int8 (Plan 1 step 2.2) ---
# Quantize the big projections; skip tiny/sensitive ones (GDN in_proj_a/b gates,
# router gate). out_proj/o_proj are RowParallel (all-reduce done by the layer's
# forward, not here). Keep bf16 weight for the high-T fallback.
_INT8_LINEAR_PREFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                         "in_proj_qkv", "in_proj_z", "out_proj")


def want_int8_linear(prefix):
    if "in_proj_a" in prefix or "in_proj_b" in prefix:
        return False
    return any(p in prefix for p in _INT8_LINEAR_PREFIXES)


def _make_linear_method():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Int8LinearMethod(UnquantizedLinearMethod):
        def __init__(self):
            super().__init__()
            self._max_t = int(os.environ.get("VLLM_INT8_LINEAR_MAX_T", "1"))

        def process_weights_after_loading(self, layer):
            w = layer.weight.data
            qi, s = quant_weight_per_outchannel(w)
            layer.weight_int8 = qi
            layer.weight_scale = s

        def apply(self, layer, x, bias=None):
            if (x.dim() == 2 and x.shape[0] <= self._max_t
                    and hasattr(layer, "weight_int8")):
                y = int8_linear_forward(x, layer.weight_int8, layer.weight_scale)
                return y if bias is None else y + bias
            return super().apply(layer, x, bias)

    return Int8LinearMethod


def get_int8_linear_method():
    return _make_linear_method()()


if __name__ == "__main__":
    dev = "xpu"; torch.manual_seed(0)
    for N, K, T in [(248320 // 2, 2048, 1), (248320 // 2, 2048, 2), (4096, 2048, 1)]:
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        qi, s = quant_weight_per_outchannel(w)
        x = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
        y = int8_linear_forward(x, qi, s); torch.xpu.synchronize()
        ref = (x.float() @ w.float().t())
        rel = (y.float() - ref).norm() / ref.norm()
        print(f"N={N} K={K} T={T}: relL2={rel:.4f} {'OK' if rel < 0.05 else 'BROKEN'}")
