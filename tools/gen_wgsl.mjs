// Regenerates shaders/ from the reference WebGPU port (../OpenDLSS-NR/ports/browser-webgpu).
//
// The port builds its two big kernels (the FP8 GEMM and the window attention) by source-to-source transforms
// in JavaScript. Running those transforms here and committing their output keeps the Rust build free of Node;
// rerun this only when the reference port changes:  node tools/gen_wgsl.mjs [path/to/browser-webgpu]
import { readFileSync, writeFileSync as writeRaw, mkdirSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const port = resolve(process.argv[2] ?? '../OpenDLSS-NR/ports/browser-webgpu');
const out = resolve('shaders');
mkdirSync(out, { recursive: true });
const writeFileSync = (path, text) =>
  writeRaw(path, typeof text === 'string' ? repeatedMirror(hardwareZero(exactNormFma(exactRounding(nagaCompatible(text))))) : text);
const load = (path) => import(pathToFileURL(join(port, path)).href);

const { variantCode } = await load('src/matmul/variants.js');
const { productionMatmulCode } = await load('src/matmul/base.js');
const { bitQuantMatmulCode } = await load('src/matmul/bit-quant.js');
const { windowAttentionCode } = await load('src/window/variants.js');
const { weightMetadataWords } = await load('src/matmul/weight-table.js');

// naga (wgpu's WGSL front end) rejects three constructs Chrome's Tint accepts. Each rewrite below keeps every
// value bit-identical; nothing else in the kernels is touched.
//
//   bitcast<vec2<f16>>(u32)        naga types this as a width-changing cast and fails; unpack2x16float is the
//                                  same reinterpretation, and f16 -> f32 -> f16 is exact.
//   bitcast<u32>(vec2<f16>(a, b))  likewise; pack2x16float of values that are already halves is exact.
//   ptr<storage> parameters        naga cannot pass a storage pointer into a function, so ops.wgsl's two
//                                  generic loaders are specialized per buffer.
function closingParen(code, open) {
  let depth = 0;
  for (let i = open; i < code.length; ++i) {
    if (code[i] === '(') depth += 1;
    else if (code[i] === ')' && --depth === 0) return i;
  }
  throw new Error('unbalanced parentheses');
}

/** Replace `call(x)` for every `call(` that starts with `prefix`, `wrap` getting the argument text x. */
function rewriteCalls(code, call, prefix, wrap) {
  for (let at = code.indexOf(prefix); at >= 0; at = code.indexOf(prefix, at + 1)) {
    const open = at + call.length - 1;
    const close = closingParen(code, open);
    const replacement = wrap(code.slice(open + 1, close));
    code = code.slice(0, at) + replacement + code.slice(close + 1);
  }
  return code;
}

function nagaCompatible(code) {
  code = rewriteCalls(code, 'bitcast<vec2<f16>>(', 'bitcast<vec2<f16>>(',
                      (x) => `vec2<f16>(unpack2x16float(${x}))`);
  code = rewriteCalls(code, 'bitcast<u32>(', 'bitcast<u32>(vec2<f16>(', (x) => `pack2x16float(vec2<f32>(${x}))`);
  const loaders = /fn (half_at|e4_at)\(buffer: ptr<storage, array<u32>, read>, index: u32\) -> f32 \{\n  return ([^\n]*);\n\}\n/g;
  const bodies = new Map([...code.matchAll(loaders)].map((m) => [m[1], m[2]]));
  if (bodies.size) {
    code = code.replace(loaders, '');
    const uses = new Set([...code.matchAll(/\b(half_at|e4_at)\(&(\w+), /g)].map((m) => `${m[1]}/${m[2]}`));
    const specialized = [...uses].sort().map((use) => {
      const [loader, buffer] = use.split('/');
      return `fn ${loader}_${buffer}(index: u32) -> f32 {\n  return ${bodies.get(loader).replace('(*buffer)', buffer)};\n}\n`;
    }).join('');
    code = code.replace(/\b(half_at|e4_at)\(&(\w+), /g, '$1_$2(');
    // After the bindings they read, before the first kernel.
    code = code.replace('\n@compute', `\n${specialized}\n@compute`);
  }
  if (/bitcast<[^>]*f16|ptr<storage/.test(code)) throw new Error('a construct naga rejects is left');
  return code;
}

// The NVIDIA Vulkan driver folds a bare f32 -> f16 -> f32 round trip, `f32(f16(x))` or its vec4 form, into
// nothing: the SPIR-V naga emits holds both conversions, but the rounding never happens. The kernels publish
// every half that way - the GEMM accumulator, and the end of every tensor-core step - so each round trip is
// replaced with the same rounding done on the bit pattern, which no compiler may remove. Round to nearest
// even, overflow to infinity, the subnormal grid below 2^-14: numerics.wgsl's f16_bits, which
// tools/check_numerics.mjs checks exhaustively against the reference. Round trips whose inner value is half
// arithmetic rather than a bare conversion (the window kernel's) are left alone; tools/verify_block0 covers them.
const EXACT_ROUND_HALF = `fn nr_round_half(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let sign = bits & 0x80000000u;
  let magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) { return value; }
  if (magnitude >= 0x477ff000u) { return bitcast<f32>(sign | 0x7f800000u); }
  if (magnitude >= 0x38800000u) {
    return bitcast<f32>(sign | ((magnitude + 0xfffu + ((magnitude >> 13u) & 1u)) & 0xffffe000u));
  }
  let exponent = magnitude >> 23u;
  if (exponent < 102u) { return bitcast<f32>(sign); }
  let mantissa = (magnitude & 0x7fffffu) | 0x800000u;
  let shift = 126u - exponent;
  var units = mantissa >> shift;
  let remainder = mantissa & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  if (remainder > halfway || (remainder == halfway && (units & 1u) != 0u)) { units += 1u; }
  return bitcast<f32>(sign | bitcast<u32>(f32(units) * 5.9604644775390625e-8));
}
fn nr_round_half4(value: vec4<f32>) -> vec4<f32> {
  // The normal half range, [2^-14, 65520), is nearly every value: there it is one vectorized mantissa
  // rounding. Anything else takes the scalar path above.
  let bits = bitcast<vec4<u32>>(value);
  let magnitude = bits & vec4<u32>(0x7fffffffu);
  if (any(magnitude < vec4<u32>(0x38800000u)) || any(magnitude >= vec4<u32>(0x477ff000u))) {
    return vec4<f32>(nr_round_half(value.x), nr_round_half(value.y), nr_round_half(value.z), nr_round_half(value.w));
  }
  let rounded = (magnitude + vec4<u32>(0xfffu) + ((magnitude >> vec4<u32>(13u)) & vec4<u32>(1u)))
    & vec4<u32>(0xffffe000u);
  return bitcast<vec4<f32>>((bits & vec4<u32>(0x80000000u)) | rounded);
}
`;

/** Replace `outer(inner(x))` by `exact(x)` wherever the inner conversion is the outer one's whole argument. */
function rewriteRoundTrips(code, outer, inner, exact) {
  let count = 0;
  for (let at = code.indexOf(outer + inner); at >= 0; at = code.indexOf(outer + inner, at + 1)) {
    if (/[\w.]/.test(code[at - 1] ?? '')) continue;
    const outerOpen = at + outer.length - 1;
    const outerClose = closingParen(code, outerOpen);
    const innerOpen = outerOpen + inner.length;
    if (closingParen(code, innerOpen) !== outerClose - 1) continue;
    code = code.slice(0, at) + `${exact}(${code.slice(innerOpen + 1, outerClose - 1)})` + code.slice(outerClose + 1);
    count += 1;
  }
  return { code, count };
}

// The window attention's cosine norm sums fma(v, v, f16(w * w)) in half. The hardware's (and native's, and upstream's
// GLSL and PTX) fma.rn.f16 rounds once; the port spells it f16(f32(a) * f32(b) + f32(c)), which rounds the sum to
// f32 first and can land it exactly on a half tie the exact sum is not on - so it rounds twice, and the other way.
// (One case in about a million norms at 2048x1152: a flipped E4M3 byte of K.) The replacement is correctly rounded:
// the product of two halves is exact in f32, TwoSum recovers the f32 sum's error exactly, and a sum that is off
// moves one f32 step towards the exact value - which cannot cross a half tie, and leaves one it sat on. The NVIDIA
// driver folds a plain TwoSum away, hence the opaque zeros below; tools/check_numerics.mjs tests this function
// against an exact oracle.
const NATIVE_NORM_FMA = 'fn nr_norm_fma(a: f16, b: f16, c: f16) -> f16 { return f16(f32(a) * f32(b) + f32(c)); }';
// Written to shaders/norm_fma.wgsl too, so tools/check_numerics.mjs tests this exact text against an exact oracle.
// `nr_opaque_zero()` is the includer's: a value that is 0 at run time but unknown when the pipeline compiles.
const NORM_FMA_CORE = `/// Whether an f32 lies exactly halfway between two adjacent halves (subnormal halves included).
fn nr_is_half_tie(value: f32) -> bool {
  let magnitude = bitcast<u32>(value) & 0x7fffffffu;
  if (magnitude >= 0x38800000u) { return (magnitude & 0x1fffu) == 0x1000u; }
  let exponent = magnitude >> 23u;
  if (exponent < 102u) { return false; }
  let mantissa = (magnitude & 0x7fffffu) | 0x800000u;
  let shift = 126u - exponent;
  return (mantissa & ((1u << shift) - 1u)) == (1u << (shift - 1u));
}

/// fma(a, b, c) in half, rounded once (fma.rn.f16). The product of two halves is exact in f32; TwoSum recovers
/// the f32 sum's error exactly. The exact value then lies strictly between the sum and its f32 neighbour towards
/// it, and no half tie can lie strictly between two adjacent f32s: so the sum rounds like the exact value unless
/// the sum is itself a tie, and then the neighbour does. The NVIDIA driver rewrites f32 algebra as if it were
/// exact ((p + c) - c into p, s - (s - c) into c), which would zero the error, so every intermediate passes
/// through an XOR with nr_opaque_zero().
fn nr_norm_fma(a: f16, b: f16, c: f16) -> f16 {
  let opaque_zero = nr_opaque_zero();
  let product = f32(a) * f32(b);
  let addend = f32(c);
  let sum = bitcast<f32>(bitcast<u32>(product + addend) ^ opaque_zero);
  let product_part = bitcast<f32>(bitcast<u32>(sum - addend) ^ opaque_zero);
  let addend_part = bitcast<f32>(bitcast<u32>(sum - product_part) ^ opaque_zero);
  let error = (product - product_part) + (addend - addend_part);
  var bits = bitcast<u32>(sum);
  if (error != 0.0 && nr_is_half_tie(sum)) { bits = select(bits - 1u, bits + 1u, (error > 0.0) == (sum > 0.0)); }
  return f16(round_f16(bitcast<f32>(bits)));
}`;
const EXACT_NORM_FMA = `/// The window kernel's opaque zero: token counts are below 2^31.
fn nr_opaque_zero() -> u32 { return attention_params.tokens >> 31u; }
${NORM_FMA_CORE}`;

function exactNormFma(code) {
  if (!code.includes('fn nr_norm_fma')) return code;
  if (!code.includes(NATIVE_NORM_FMA)) throw new Error('nr_norm_fma changed upstream; revisit the rewrite');
  return code.replace(NATIVE_NORM_FMA, EXACT_NORM_FMA);
}

// The f16 tensor-core step publishes a zero result as +0 even when its fixed-point sum was a negative value too
// small for a half (tools/mma_check: -2^-30 -> 0x0000 on the hardware, 0x8000 in the port's and the CPU
// reference's model). The port's FP8 GEMM already publishes zero as +0; its f16 step, used by the input adapter
// and the head, did not, which left a -0 in the head where native has +0.
const SIGNED_SUBNORMAL_ZERO = '    half_bits = half_bits | min(mantissa, 1024u);';
const UNSIGNED_SUBNORMAL_ZERO = `    if (mantissa == 0u) { return 0.0; }   // the hardware publishes a zero as +0
    half_bits = half_bits | min(mantissa, 1024u);`;

function hardwareZero(code) {
  if (!code.includes('fn fixed_to_f16')) return code;
  if (!code.includes(SIGNED_SUBNORMAL_ZERO)) throw new Error('fixed_to_f16 changed upstream; revisit the rewrite');
  return code.replace(SIGNED_SUBNORMAL_ZERO, UNSIGNED_SUBNORMAL_ZERO);
}

// The padded field is at least 320 on each axis, so an image under ~161 pixels has more padding than one
// reflection covers, and upstream's `2 valid - x - 2` (here and in its GLSL) wraps around as an unsigned integer:
// an out-of-bounds read, which WGSL clamps and CUDA faults on. The mirror continues instead, with period
// 2 (valid - 1): identical to upstream wherever upstream's index is in range. (Native behaviour for such small
// images is not known; the fixtures are far larger.)
const MIRROR_FN = `/// Reflect-101 (no edge repeat), repeated for padding wider than the image.
fn nr_mirror(i: u32, n: u32) -> u32 {
  if (i < n) { return i; }
  if (n == 1u) { return 0u; }
  let period = 2u * (n - 1u);
  let r = i % period;
  return select(period - r, r, r < n);
}
`;

function repeatedMirror(code) {
  const x = 'select(2u * pre.valid_width - id.x - 2u, id.x, id.x < pre.valid_width)';
  const y = 'select(2u * pre.valid_height - id.y - 2u, id.y, id.y < pre.valid_height)';
  if (!code.includes(x)) return code;
  if (!code.includes(y)) throw new Error('preprocess.wgsl changed upstream; revisit the mirror rewrite');
  code = code.replace(x, 'nr_mirror(id.x, pre.valid_width)').replace(y, 'nr_mirror(id.y, pre.valid_height)');
  return code.replace('@compute @workgroup_size(8, 8)\nfn preprocess', `${MIRROR_FN}\n@compute @workgroup_size(8, 8)\nfn preprocess`);
}

function exactRounding(code) {
  const scalar = rewriteRoundTrips(code, 'f32(', 'f16(', 'nr_round_half');
  const vector = rewriteRoundTrips(scalar.code, 'vec4<f32>(', 'vec4<f16>(', 'nr_round_half4');
  if (!scalar.count && !vector.count) return code;
  // After the `enable` directive, which has to come first.
  return vector.code.replace(/^(enable f16;\n)/m, `$1${EXACT_ROUND_HALF}`);
}

const header = (what) => `// GENERATED by tools/gen_wgsl.mjs from the OpenDLSS-NR browser-webgpu port: ${what}. Do not edit.\n`;
const numerics = readFileSync(join(port, 'shaders/numerics.wgsl'), 'utf8');

// src/graph.js only ever asks for the packed-residual variant: an f16 skip is told apart at run time by the
// FLAG_RESIDUAL_E4 bit, not by the variant. So the variants are output format x batched.
for (const output of ['e4', 'half', 'dual']) {
  for (const batched of [false, true]) {
    const name = `gemm_${output}${batched ? '_batched' : ''}.wgsl`;
    writeFileSync(join(out, name), header(`FP8 GEMM ${name}`) +
      variantCode({ output, residual: 'e4', batched, tile128: true }));
  }
}

writeFileSync(join(out, 'window_attention.wgsl'),
  header('shifted-window attention') + ['enable f16;', numerics, windowAttentionCode()].join('\n'));

// The two lookup tables the GEMM reads, built on the device from the retained activation arithmetic
// (src/matmul/silu-table.js and packed-activation.js).
const code = bitQuantMatmulCode(productionMatmulCode());
const fn = (name) => {
  const begin = code.indexOf(`fn ${name}(`), end = code.indexOf('\n}', begin) + 2;
  if (begin < 0 || end < 2) throw new Error(`missing ${name}`);
  return code.slice(begin, end);
};
writeFileSync(join(out, 'silu_table.wgsl'), header('half SiLU table builder') + `enable f16;
fn round_accumulator(value: f32) -> f32 { return f32(f16(value)); }
${fn('mp_cubic_silu')}
${fn('fp8_domain')}
@group(0) @binding(0) var<storage, read_write> values: array<vec2<f16>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let value = f32(bitcast<vec2<f16>>(id.x).x);
  let activated = mp_cubic_silu(value);
  values[id.x] = vec2<f16>(f16(activated), f16(fp8_domain(activated)));
}
`);
const packed = readFileSync(join(port, 'src/matmul/packed-activation.js'), 'utf8');
const exact = packed.match(/const exactCode = `([\s\S]*?)`;/)[1];
writeFileSync(join(out, 'packed_silu_table.wgsl'), header('packed E4 SiLU table builder') + `enable f16;
${exact}
@group(0) @binding(0) var<storage, read> source: array<vec2<f16>>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  var word = 0u;
  for (var part = 0u; part < 4u; part++) {
    word |= exact_e4_output_code(f32(source[id.x * 4u + part].y)) << (part * 8u);
  }
  result[id.x] = word;
}
`);
writeFileSync(join(out, 'weight_metadata.bin'), Buffer.from(weightMetadataWords().buffer));

writeRaw(join(out, 'norm_fma.wgsl'), header('the window norm half fma, for tools/check_numerics.mjs') + NORM_FMA_CORE + '\n');
for (const file of ['numerics.wgsl', 'gemm_f16.wgsl', 'vit.wgsl', 'ops.wgsl', 'preprocess.wgsl', 'selftest.wgsl']) {
  writeFileSync(join(out, file), readFileSync(join(port, 'shaders', file), 'utf8'));
}
console.log(`wrote ${out}`);
