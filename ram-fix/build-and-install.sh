#!/bin/bash
# Build + install the xe "force_p2p_vram" patch that stops the Intel xe driver from mirroring the
# entire multi-GPU VRAM working set into host RAM (see UPSTREAM_REPORT.md). Reclaims ~the whole shadow
# (e.g. 4-card bf16 ~121 GiB host -> ~22 GiB), capability-neutral. HOST-side, root, kernel-module change
# — it cannot ride in the docker image (containers share the host kernel). Default is byte-identical
# stock behavior unless force_p2p_vram=1 is set (which this installer does).
#
# Usage:   sudo ./build-and-install.sh [KERNEL_SRC_DIR]
#   - no arg: fetches the matching mainline source for a vX.Y-rcZ / vX.Y kernel from GitHub.
#   - KERNEL_SRC_DIR: point at your own configured kernel source tree (REQUIRED for distro kernels,
#     e.g. Ubuntu HWE — install its linux-source package and pass its path).
# Revert anytime with ./uninstall.sh . SAFETY: only enable where device-to-device PCIe P2P works
# (any modern multi-GPU board on a single CPU socket; verify your GPUs can P2P before trusting it).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
KREL="$(uname -r)"
BUILD="/lib/modules/$KREL/build"
PATCH="$HERE/xe-p2p.patch"
ts(){ date "+%H:%M:%S"; }
say(){ echo "[$(ts)] $*"; }

[ "$(id -u)" = "0" ] || { echo "run as root (sudo)"; exit 1; }
[ -f "$PATCH" ] || { echo "missing $PATCH"; exit 1; }
modinfo xe >/dev/null 2>&1 || { echo "xe is not a loadable module on this kernel; can't hot-patch."; exit 1; }
[ -f "$BUILD/Module.symvers" ] || { echo "missing kernel build tree/headers ($BUILD). Install linux-headers-$KREL."; exit 1; }

# ---- 1. obtain a kernel source tree matching the running kernel ----
KSRC="${1:-}"
if [ -z "$KSRC" ]; then
  # derive a mainline tag from uname, e.g. 7.1.0-070100rc7-generic -> v7.1-rc7 ; 6.14.2-... -> v6.14.2
  BASE="${KREL%%-*}"                 # 7.1.0
  RC="$(echo "$KREL" | grep -oE 'rc[0-9]+' | head -1)"
  MAJMIN="$(echo "$BASE" | cut -d. -f1-2)"; SUB="$(echo "$BASE" | cut -d. -f3)"
  if [ -n "$RC" ]; then TAG="v${MAJMIN}-${RC}"; elif [ "${SUB:-0}" = "0" ]; then TAG="v${MAJMIN}"; else TAG="v${BASE}"; fi
  say "no source dir given; guessing mainline tag $TAG for kernel $KREL"
  KSRC="$HERE/linux-src"
  if [ ! -d "$KSRC" ]; then
    say "downloading $TAG source (~250 MB)..."
    curl -fSL "https://github.com/torvalds/linux/archive/refs/tags/${TAG}.tar.gz" -o "$HERE/linux-src.tgz"
    mkdir -p "$KSRC"; tar xzf "$HERE/linux-src.tgz" -C "$KSRC" --strip-components=1
  fi
  echo ""
  echo "NOTE: if you run a DISTRO kernel (Ubuntu/Fedora/etc), the mainline tag will NOT match — the"
  echo "      module will fail to load with vermagic/CRC errors. Install your distro's kernel source"
  echo "      package and re-run:  sudo ./build-and-install.sh /path/to/your/kernel/source"
  echo ""
fi
[ -d "$KSRC/drivers/gpu/drm/xe" ] || { echo "no xe driver source under $KSRC"; exit 1; }

# ---- 2. configure the tree to match the running kernel exactly (vermagic + CRCs) ----
cd "$KSRC"
say "configuring to match $KREL ..."
cp "/boot/config-$KREL" .config
# make `make kernelrelease` == uname -r so the module vermagic matches
sed -i 's/^EXTRAVERSION =.*/EXTRAVERSION =/' Makefile
LOCALVER="-${KREL#*-}"              # e.g. -070100rc7-generic
./scripts/config --disable LOCALVERSION_AUTO
./scripts/config --set-str LOCALVERSION "$LOCALVER"
./scripts/config --disable MODULE_SIG_FORCE 2>/dev/null || true
make olddefconfig >/dev/null 2>&1
KR="$(make -s kernelrelease)"
[ "$KR" = "$KREL" ] || say "WARNING: kernelrelease '$KR' != running '$KREL' (module may refuse to load)"

say "modules_prepare (needs: flex bison libelf-dev libssl-dev libdw-dev bc) ..."
make -j"$(nproc)" modules_prepare
cp "$BUILD/Module.symvers" ./Module.symvers          # real CRCs so the module loads into THIS kernel

# ---- 3. apply the patch + build just xe.ko ----
say "applying xe-p2p.patch ..."
patch -p1 --forward --reject-file=- < "$PATCH" || { echo "patch failed (already applied? wrong kernel?)"; exit 1; }
say "building xe.ko (compiles xe + display objects; several minutes) ..."
XEDIR="$PWD/drivers/gpu/drm/xe"
PATH="$XEDIR:$XEDIR/generated:$PATH" make -j"$(nproc)" KBUILD_MODPOST_WARN=1 M=drivers/gpu/drm/xe modules
[ -f drivers/gpu/drm/xe/xe.ko ] || { echo "build produced no xe.ko"; exit 1; }
modinfo drivers/gpu/drm/xe/xe.ko | grep -q force_p2p_vram || { echo "built module lacks force_p2p_vram param"; exit 1; }

# ---- 4. install (updates/ overrides the stock module) + enable the param ----
say "installing to /lib/modules/$KREL/updates/xe.ko + modprobe.d ..."
install -D -m0644 drivers/gpu/drm/xe/xe.ko "/lib/modules/$KREL/updates/xe.ko"
depmod -a "$KREL"
echo "options xe force_p2p_vram=1" > /etc/modprobe.d/xe-p2p.conf

say "DONE. modinfo now: $(modinfo xe | awk '/^filename/{print $2}')"
echo ""
echo "Apply now WITHOUT reboot (stop all GPU workloads first — this unloads the driver):"
echo "  for d in \$(ls /sys/bus/pci/drivers/xe/ | grep '^0000:'); do echo \$d | sudo tee /sys/bus/pci/drivers/xe/unbind; done"
echo "  sudo rmmod xe && sudo modprobe xe"
echo "  cat /sys/module/xe/parameters/force_p2p_vram   # expect Y"
echo "Or just reboot. Verify the win: serve a multi-GPU config and watch 'free -g' — host RAM should"
echo "drop from ~(sum of per-card VRAM) to just the local footprint. Revert: sudo ./uninstall.sh"
