# Optional host-side RAM fix — reclaim the "VRAM shadow"

**TL;DR:** on multi-GPU (tensor-parallel) serving, the Intel `xe` driver mirrors the *entire* combined
VRAM working set into **host RAM** — ~72 GiB for a 2-card run, ~121 GiB for 4 cards — invisible to
normal tools and released only when the GPU workload stops. This is a one-function inefficiency in
`xe`'s dma-buf export path. The patch here fixes it. Measured on Qwen3.6-35B-A3B:

| config | host RAM (stock) | host RAM (patched) | capability |
|---|---|---|---|
| int8-tp2 (2 cards) | 72 GiB | **14 GiB** | unchanged |
| bf16-tp4 (4 cards) | 121 GiB | **22 GiB** | unchanged |

It reclaims ~the whole shadow with **no measured capability change** (MMLU-Redux / IFEval / GSM8K all
land on reference — the reads still go over PCIe P2P, they just stop being duplicated in system RAM).

## Why this is NOT in the docker image

Containers share the **host kernel**. `xe` is a host kernel module; a container can't ship or load one.
It's also locked to your exact kernel version. So this is inherently a **host-side, root, one-time**
step, separate from `docker pull`. Everything else in this repo (vLLM, kernels, `serve.py`) is userspace
and needs none of this — apply this **only if** you want the host RAM back.

## What it does

Root cause (full writeup in [UPSTREAM_REPORT.md](UPSTREAM_REPORT.md)): `xe_gem_prime_export()` calls
`ttm_bo_setup_export()` → `ttm_tt_populate()`, which allocates a **full-size system-memory copy** of
every exported buffer, even though the buffer stays in VRAM and peers read it via PCIe P2P. The patch
adds a module parameter **`force_p2p_vram`** that skips that population. Default **off** = byte-identical
stock behavior; the installer turns it **on**.

## Requirements

- `xe` as a **loadable module** (`modinfo xe` works), kernel headers for your running kernel, and a
  kernel **source tree that matches your kernel** (see below).
- Build deps: `flex bison libelf-dev libssl-dev libdw-dev bc` and a compiler matching your kernel's.
- A board where device-to-device **PCIe P2P actually works** (any modern multi-GPU setup on one CPU
  socket — this is what the all-reduce already relies on). Don't force it if P2P is broken on your box.

## Install

```bash
# Mainline / mainline-PPA kernel (e.g. Ubuntu mainline vX.Y-rcZ): auto-fetches matching source
sudo ./build-and-install.sh

# DISTRO kernel (Ubuntu HWE, Fedora, etc): the mainline tag won't match — install your distro's
# kernel source package and pass its path so vermagic/CRCs line up:
sudo ./build-and-install.sh /usr/src/linux-source-<your-version>
```

Then apply without a reboot (stop GPU workloads first — it unloads the driver):

```bash
for d in $(ls /sys/bus/pci/drivers/xe/ | grep '^0000:'); do echo $d | sudo tee /sys/bus/pci/drivers/xe/unbind; done
sudo rmmod xe && sudo modprobe xe
cat /sys/module/xe/parameters/force_p2p_vram      # expect: Y
```

…or just reboot. Verify: serve a multi-GPU config and watch `free -g` — host "used" should drop from
roughly the sum of per-card VRAM to just the local footprint.

**Revert:** `sudo ./uninstall.sh` (removes the module + param; stock `xe` loads on reload/reboot).

## Caveats / honesty

- **Console safety:** unloading `xe` is safe *if your console/display isn't on an xe GPU* (e.g. a BMC/ast
  display or a separate card). If your only display is an Arc GPU, do the install and reload from SSH,
  or just reboot.
- **Kernel updates:** this installs into `updates/`, which a kernel upgrade will orphan (you'd silently
  be back to the big-RAM behavior). Re-run the installer after a kernel update. (A DKMS package that
  auto-rebuilds is the nicer long-term answer, but `xe` being an in-tree driver with host-tool/display
  build coupling makes a clean DKMS non-trivial — not shipped yet.)
- **Validation scope:** verified on 2× and 4× Arc Pro B70, Qwen3.6-35B-A3B, kernel 7.1-rc7. Correctness
  checked with the full capability suite (no change). Your mileage on other kernels/GPUs may vary.
- **This is headed upstream.** The real fix is Intel making the export populate lazy/P2P-aware in `xe`
  itself; see UPSTREAM_REPORT.md. If that lands, you won't need this at all.
