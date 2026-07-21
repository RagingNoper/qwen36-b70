# xe: dma-buf prime export populates full-size system pages for VRAM BOs even when importers use PCIe P2P

## Summary

On a multi-GPU (tensor-parallel) inference workload, host system RAM usage grows to
approximately the **total VRAM working set of all GPUs** (e.g. ~72 GiB for a 2-card run,
~120+ GiB for 4 cards), even though every buffer is VRAM-resident and peers read each
other's buffers over PCIe **peer-to-peer (P2P)**. This host-RAM "shadow" is not visible in
process RSS and is released the moment the GPU workload stops.

Root cause: **`xe_gem_prime_export()` unconditionally calls `ttm_bo_setup_export()`, which
populates a full-size system-memory (`ttm_tt`) backing for the exported buffer object** —
even when the importer is P2P-capable and never touches those system pages.

## Environment

- Kernel: Linux 7.1-rc7 (`drivers/gpu/drm/xe`), `xe` as a module.
- GPUs: 4× Intel Arc Pro B70 (Battlemage / BMG), 32 GiB each.
- Userspace: Level-Zero / NEO compute-runtime; tensor-parallel LLM inference (vLLM-XPU)
  that shares per-rank buffers across cards via L0 IPC (`zeMemGetIpcHandle` /
  `zeMemOpenIpcHandle`), which lands on the kernel dma-buf prime export/import path.
- `CONFIG_PCI_P2PDMA=y`; device-to-device PCIe P2P is functional on this platform
  (verified: cross-card P2P bandwidth matrix is healthy).

## Symptom

Serving one 35B-parameter MoE model tensor-parallel across the cards:

| cards | host RAM used (serving) | GPU working set |
|------:|------------------------:|----------------:|
| 2 (TP2) | ~72 GiB | ~2× ~28 GiB VRAM |
| 4 (TP4) | ~120+ GiB | ~4× VRAM |

The host RAM is not RSS, page cache, slab, shmem, or tmpfs — it is GPU-driver host-side
allocation, and it scales with the number of cards / the shared working set.

## Root cause / path

Instrumented counters (module params added to `xe_dma_buf.c`) over one 2-card serve:

```
xe_dma_buf_pin        = 0     # importers use DYNAMIC attach (move_notify), never .pin
xe_gem_prime_export   = 806   # -> ttm_bo_setup_export -> ttm_bo_populate -> ttm_tt_populate
xe_dma_buf_create_obj = 818   # cross-device imports (XE_BO_FLAG_SYSTEM sg BOs)
xe_dma_buf_map VRAM   = 554   # importer reads via P2P VRAM sgt
xe_dma_buf_map TT     = 8     # importer reads via system-pages sgt
```

`ttm_bo_setup_export()` → `ttm_bo_populate()` → `ttm_tt_populate()` allocates the BO's full
`ttm_tt` system pages. It runs for **every** exported BO (806×), yet the importer reads via
the VRAM P2P sgt path 554/562 times — the system-page backing is **redundant** for the P2P
consumers. The populate is presumably defensive (so an importer that maps into `XE_PL_TT`
has CPU-accessible pages), but on a P2P-capable topology it materializes the entire
cross-GPU working set a second time in host RAM.

`drivers/gpu/drm/xe/xe_dma_buf.c`, `xe_gem_prime_export()`:

```c
        ret = ttm_bo_setup_export(&bo->ttm, &ctx);   /* populates full system ttm_tt */
        if (ret)
                goto out_put;
```

## Evidence the population is the cause

Gating that single call behind a module param and skipping it when P2P is expected:

```c
        if (!xe_force_p2p_vram) {
                ret = ttm_bo_setup_export(&bo->ttm, &ctx);
                if (ret)
                        goto out_put;
        }
```

Result on the same 2-card serve (int8-tp2): **host RAM 72 GiB → 14 GiB**, model serves
normally (custom all-reduce, cudagraphs, correctness all intact — GSM8K 93.3% unchanged).

## Proposed direction (not a final patch)

The unconditional full-size system populate on export is wasteful whenever the importer can
(and does) use P2P. Options for upstream discussion:

1. **Lazy-populate:** don't populate system pages at export time; only populate on the
   `map_dma_buf` path that actually needs a `XE_PL_TT` sgt (the map already migrates/validates
   there). Exports consumed purely via P2P would then never allocate the shadow.
2. **P2P-aware:** skip the export-time populate when the exporting BO is VRAM-resident and the
   platform/importer advertises P2P (the attach path already negotiates `peer2peer`).
3. At minimum, a documented knob, since on large multi-GPU boxes this doubles host-RAM
   requirements for no functional benefit.

## Reproduce

Multi-GPU tensor-parallel workload that shares device buffers across cards via L0 IPC;
observe `free` host-used ≈ Σ per-card VRAM working set while serving, released on stop.
The `dma_buf/bufinfo` total stays the same with/without the fix (nominal buffer sizes),
but actual host RAM collapses to the local (non-exported) footprint.
