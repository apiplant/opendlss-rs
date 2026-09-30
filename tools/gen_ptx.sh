#!/bin/sh
# Regenerates ptx/ from the reference's PTX emitters (../OpenDLSS-NR/scripts/ptx), with exactly the variant set of
# its scripts/build_shaders.ps1. Committed, so building needs no Python; rerun when the reference changes:
#   tools/gen_ptx.sh [path/to/OpenDLSS-NR]
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
ref=$(cd "${1:-$root/../OpenDLSS-NR}" && pwd)
gen="$ref/scripts/ptx"
out="$root/ptx"
mkdir -p "$out"
run() { script=$1; shift; python3 "$gen/$script.py" "$@" >/dev/null; }

for k in 64 128 256; do run mlp_e4m3 $k "$out/mlp_e4m3_K$k.ptx" 3 1 1; done
for c in 64 128 256 512; do run qkv_e4m3 $c "$out/qkv_e4m3_K$c.ptx" 80; done
for k in 32 64 128 256 512 1024; do
  for f in 5 13 4 8 6; do run gemm2_e4m3 $k $f "$out/gemm2_e4m3_K${k}_f$f.ptx"; done
done
for cfg in "64 4 0" "128 4 0" "256 3 80" "256 4 64"; do
  set -- $cfg
  run ffn_e4m3 $1 $2 "$out/ffn_e4m3_C$1_R$2.ptx" $3
  run ffn_e4m3 $1 $2 "$out/ffn_e4m3_C$1_R$2_proj.ptx" $3 1
done
for cfg in "4096 1 32 4" "1024 0 16 2" "1024 1 8 4" "1024 0 8 4"; do
  set -- $cfg
  run gemmt_e4m3 $1 $2 $3 $4 "$out/gemmt_e4m3_K$1_f$2_p$3_s$4.ptx"
done
for cfg in "4 4" "2 8" "4 8"; do
  set -- $cfg
  run reduce_e4m3 $1 $2 "$out/reduce_e4m3_s$1_f$2.ptx"
done
for cfg in "1024 6 1" "4096 5 4" "1024 8 2" "1024 5 4"; do
  set -- $cfg
  run gemmv_e4m3 $1 $2 $3 "$out/gemmv_e4m3_K$1_f$2_s$3.ptx"
  run gemmv_e4m3 $1 $2 $3 "$out/gemmv_e4m3_K$1_f$2_s$3_m96.ptx" 4 6 0 0 0 96
done
for padded in 64 128 192 256; do run global_attention_e4m3 $padded "$out/global_attention_e4m3_p$padded.ptx"; done
run global_attention_stream_e4m3 normalize "$out/global_normalize_e4m3.ptx"
run global_attention_stream_e4m3 attention "$out/global_attention_stream_e4m3.ptx" 4
for cfg in "512 4" "512 5" "512 13" "256 5" "4096 5" "1024 8" "1024 5"; do
  set -- $cfg
  run gemmv_e4m3 $1 $2 1 "$out/gemmv_e4m3_K$1_f$2_s1_m96.ptx" 4 6 0 0 0 96
done
for f in 2 66 74 130 48; do run block32_e4m3 $f "$out/block32_e4m3_f$f.ptx"; done
# The 512 stage's branch MLP as two batched launches (tools/ptx/gemm2_batched.py: upstream's gemm2 with grid z
# selecting the branch).
python3 "$root/tools/ptx/gemm2_batched.py" "$gen" 64 6 "$out/gemm2b_e4m3_K64_f6.ptx" >/dev/null
python3 "$root/tools/ptx/gemm2_batched.py" "$gen" 256 4 "$out/gemm2b_e4m3_K256_f4.ptx" >/dev/null
# The kernels of the fast route that exist only as GLSL upstream, in CUDA C (cuda/ops.cu). -fmad=false: no
# multiply-add is contracted; every rounding is where the source puts it.
"${NVCC:-nvcc}" -ptx -arch=sm_89 -O3 -fmad=false -o "$out/ops_cuda.ptx" "$root/cuda/ops.cu"
echo "wrote $(ls "$out" | wc -l) PTX files to $out"
