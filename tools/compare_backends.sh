#!/bin/sh
# Runs each image through both backends and checks that the heads (RGB residual and blend logit, f32) are
# bit-identical: the wgpu route emulates the arithmetic in WGSL, the CUDA route runs upstream's PTX on the tensor
# cores, so agreement is a check of both. Extra arguments go to `process` (e.g. --seed 7 --style 2).
#   tools/compare_backends.sh image... [-- process options]
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
bin="$root/target/release/opendlss-nr"
out=${TMPDIR:-/tmp}/opendlss-compare-backends
mkdir -p "$out"
images=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do images="$images $1"; shift; done
[ "$1" = "--" ] && shift
status=0
for image in $images; do
  name=$(basename "$image")
  "$bin" process --backend wgpu -i "$image" -o "$out/$name.wgpu.png" --head "$out/$name.wgpu.head" "$@" >/dev/null
  "$bin" process --backend cuda -i "$image" -o "$out/$name.cuda.png" --head "$out/$name.cuda.head" "$@" >/dev/null
  if cmp -s "$out/$name.wgpu.head" "$out/$name.cuda.head"; then
    echo "identical  $name $*"
  else
    echo "DIFFERENT  $name $*"; status=1
  fi
done
exit $status
