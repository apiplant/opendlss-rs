#!/bin/sh
# Builds the harness against the reference sources and checks block 0 of one frame of <image>.
#   tools/verify_block0/run.sh <image> [rows]      (OPENDLSS_NR_REF defaults to ../OpenDLSS-NR)
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
ref=${OPENDLSS_NR_REF:-$root/../OpenDLSS-NR}
out=${TMPDIR:-/tmp}/opendlss-verify-block0
mkdir -p "$out/src"
# reference.h includes "nr_model.h" by quotes, which resolves beside it first; build from copies placed beside
# the Vulkan-free stand-in instead.
cp "$ref/src/reference.cpp" "$ref/src/reference.h" "$ref/src/numeric.h" "$ref/src/json.h" "$here/shim/nr_model.h" "$out/src/"
# Upstream's norm fma rounds twice (see shim/nr_model.h); the copy uses the correctly rounded one.
line='r[c] = roundF16((float)((double)x[c] * (double)x[c] + (double)highSquare));'
grep -qF "$line" "$out/src/reference.cpp" || { echo "reference.cpp's norm fma changed upstream; revisit the patch" >&2; exit 1; }
sed -i "s|r\[c\] = roundF16((float)((double)x\[c\] \* (double)x\[c\] + (double)highSquare));|r[c] = halfFmaOnce(x[c], highSquare, roundF16);|" "$out/src/reference.cpp"
c++ -std=c++20 -O2 -I"$out/src" "$here/verify_block0.cpp" "$out/src/reference.cpp" -o "$out/verify_block0"
c++ -std=c++20 -O2 -I"$out/src" "$here/verify_expert_block.cpp" "$out/src/reference.cpp" -o "$out/verify_expert_block"
labels="input features,adapter f16,adapter e4,full ffn,full ffn residual,full ffn quantized,full qkv,full attended,block 0 raw,block 0 out,post qkv,post attended"
field=$("$root/target/release/opendlss-nr" dump-tensors --model "$root/models/nr" -i "$1" --out "$out" --tensors "$labels" | sed -n 's/^field //p')
"$out/verify_block0" "$root/models/nr" "$out" "${field%x*}" "${field#*x}" "${2:-4096}"
