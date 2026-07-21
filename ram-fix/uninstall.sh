#!/bin/bash
# Revert the xe force_p2p_vram patch: remove the patched module + param so the stock in-tree xe loads.
set -eu
KREL="$(uname -r)"
[ "$(id -u)" = "0" ] || { echo "run as root (sudo)"; exit 1; }
rm -f "/lib/modules/$KREL/updates/xe.ko" /etc/modprobe.d/xe-p2p.conf
depmod -a "$KREL"
echo "removed patched module + param. Now resolving to: $(modinfo xe | awk '/^filename/{print $2}')"
echo "Apply now without reboot: stop GPU workloads, unbind the cards, then 'sudo rmmod xe && sudo modprobe xe'."
echo "(Or reboot — the stock xe will load.)"
