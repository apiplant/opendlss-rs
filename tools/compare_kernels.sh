#!/bin/sh
# Runs one frame of <image> with this crate's rewritten kernels and with the reference port's, and checks
# that the head (and so every tensor before it) is byte-identical.
#   tools/compare_kernels.sh <image>
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
out=${TMPDIR:-/tmp}/opendlss-compare-kernels
mkdir -p "$out"
bin="$root/target/release/opendlss-nr"
tensors="vit attended,vit state,head"
NR_REFERENCE_KERNELS=1 "$bin" dump-tensors --model "$root/models/nr" -i "$1" --out "$out/reference" --tensors "$tensors" >/dev/null
"$bin" dump-tensors --model "$root/models/nr" -i "$1" --out "$out/fast" --tensors "$tensors" >/dev/null
status=0
for t in "vit attended" "vit state" "head"; do
  if cmp -s "$out/reference/$t.bin" "$out/fast/$t.bin"; then echo "identical  $t"; else echo "DIFFERENT  $t"; status=1; fi
done
exit $status
