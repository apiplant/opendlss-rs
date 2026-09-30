#!/bin/sh
# Hardware FP8 MMA vs the reference's model of it. OPENDLSS_NR_REF defaults to ../OpenDLSS-NR.
set -e
here=$(cd "$(dirname "$0")" && pwd)
ref=${OPENDLSS_NR_REF:-$here/../../../OpenDLSS-NR}
out=${TMPDIR:-/tmp}/opendlss-mma-check
mkdir -p "$out/src"
cp "$ref/src/reference.cpp" "$ref/src/reference.h" "$ref/src/numeric.h" "$here/../verify_block0/shim/nr_model.h" "$out/src/"
"${NVCC:-nvcc}" -O2 -std=c++20 -arch=sm_89 -I"$out/src" "$here/mma_check.cu" "$out/src/reference.cpp" -o "$out/mma_check"
"$out/mma_check" "$@"
