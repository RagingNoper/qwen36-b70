"""Batch-1-optimized int8 W8A8 MoE GEMV for the NATIVE vLLM FusedMoE path (XPU).

Replaces the Triton *grouped* GEMM at decode (1 token x topk experts = ~1 row per
expert = occupancy-starved) with direct expert-indexed streaming GEMVs. Uses the
native out-major weight layout (w13[E,2I,K], w2[E,H,I]) whose contiguous per-
output-channel rows are ideal for a GEMV. Fused per-token/per-slot int8 activation
quant in each kernel prologue (2 launches, no extra quant kernels). Weights are the
online-quantized int8 params (per-out-channel scales w13_scale[E,2I], w2_scale[E,H]).
"""
import torch
import triton
import triton.language as tl


def _np2(x):
    return 1 << (x - 1).bit_length()


# per-shape (BN, BK, W); fall back to defaults. TP2 gate_up I=256, down H=2048.
# Tuned via min-of-N do_bench at the TP2 T=1 decode shape (tune_native_gemv.py):
# gate_up 407 GB/s (72% roof), down 234 GB/s (42% roof).
_GU = {256: (32, 128, 16), 128: (8, 256, 8)}
_DN = {2048: (8, 128, 8)}


@triton.jit
def _gu(x, w13, ws13, ids, h,
        K: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr, BLK: tl.constexpr):
    slot = tl.program_id(0); nb = tl.program_id(1)
    t = slot // TOPK; e = tl.load(ids + slot).to(tl.int64)
    on = nb * BN + tl.arange(0, BN)                       # over I (gate/up out chan)
    full = tl.arange(0, BLK); mk = full < K
    xr = tl.load(x + t * K + full, mask=mk, other=0.0).to(tl.float32)
    xs = tl.maximum(tl.max(tl.abs(xr), 0) / 127.0, 1e-8); inv = 1.0 / xs
    ag = tl.zeros((BN,), tl.int32); au = tl.zeros((BN,), tl.int32)
    wg = w13 + e * (2 * I) * K + on[:, None] * K         # gate rows [BN,K] (contiguous over K)
    wu = w13 + e * (2 * I) * K + (on[:, None] + I) * K    # up rows
    for k0 in range(0, K, BK):
        ok = k0 + tl.arange(0, BK)
        xf = tl.load(x + t * K + ok).to(tl.float32)
        xq = tl.minimum(tl.maximum(tl.floor(xf * inv + 0.5), -127.0), 127.0).to(tl.int32)
        ag += tl.sum(tl.load(wg + ok[None, :]).to(tl.int32) * xq[None, :], 1)
        au += tl.sum(tl.load(wu + ok[None, :]).to(tl.int32) * xq[None, :], 1)
    sg = tl.load(ws13 + e * (2 * I) + on); su = tl.load(ws13 + e * (2 * I) + on + I)
    gf = ag.to(tl.float32) * xs * sg; uf = au.to(tl.float32) * xs * su
    tl.store(h + slot * I + on, (gf * tl.sigmoid(gf) * uf).to(tl.bfloat16))


@triton.jit
def _dn(h, w2, ws2, ids, tw, out,
        I: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr, BLK: tl.constexpr):
    t = tl.program_id(0); nb = tl.program_id(1)
    on = nb * BN + tl.arange(0, BN)                       # over H
    acc = tl.zeros((BN,), tl.float32)
    for j in range(TOPK):
        slot = t * TOPK + j
        e = tl.load(ids + slot).to(tl.int64)
        wgt = tl.load(tw + slot).to(tl.float32)
        full = tl.arange(0, BLK); mk = full < I
        hr = tl.load(h + slot * I + full, mask=mk, other=0.0).to(tl.float32)
        hsc = tl.maximum(tl.max(tl.abs(hr), 0) / 127.0, 1e-8); inv = 1.0 / hsc
        pi = tl.zeros((BN,), tl.int32)
        wb = w2 + e * H * I + on[:, None] * I             # w2[e, n(H), i] rows [BN,I]
        for k0 in range(0, I, BK):
            ok = k0 + tl.arange(0, BK)
            hf = tl.load(h + slot * I + ok).to(tl.float32)
            hq = tl.minimum(tl.maximum(tl.floor(hf * inv + 0.5), -127.0), 127.0).to(tl.int32)
            pi += tl.sum(tl.load(wb + ok[None, :]).to(tl.int32) * hq[None, :], 1)
        sw = tl.load(ws2 + e * H + on)
        acc += wgt * (pi.to(tl.float32) * hsc * sw)
    tl.store(out + t * H + on, acc.to(tl.bfloat16))


def moe_int8_gemv(x, w13, ws13, w2, ws2, topk_weights, topk_ids, h_buf, out_buf):
    """x:[T,K] bf16; w13:[E,2I,K] int8, ws13:[E,2I]; w2:[E,H,I] int8, ws2:[E,H]."""
    T, K = x.shape
    E, twoI, _ = w13.shape
    I = twoI // 2
    H = w2.shape[1]
    TOPK = topk_ids.shape[-1]
    ids = topk_ids.reshape(-1).contiguous()
    tw = topk_weights.reshape(-1).to(torch.float32).contiguous()
    gn, gk, gw = _GU.get(I, (8, 256, 8))
    dn, dk, dw = _DN.get(H, (16, 128, 8))
    _gu[(T * TOPK, triton.cdiv(I, gn))](x, w13, ws13, ids, h_buf,
        K=K, I=I, TOPK=TOPK, BN=gn, BK=gk, BLK=_np2(K), num_warps=gw)
    _dn[(T, triton.cdiv(H, dn))](h_buf, w2, ws2, ids, tw, out_buf,
        I=I, H=H, TOPK=TOPK, BN=dn, BK=dk, BLK=_np2(I), num_warps=dw)
    return out_buf[:T]
