// HAND-WRITTEN (not generated): the ViT attention of vit.wgsl, rearranged to keep the workgroup busy.
//
// vit.wgsl's `vit_attend` gives each (head, query) a workgroup of 64 threads, sums all PADDED_TOKENS
// exponentials on one of them, and runs the value reduction on 32. This kernel changes only who computes
// what, never what is computed:
//
//   * a workgroup takes two queries, so the 32-component value reduction - a chain over the keys that the
//     reference defines as sequential - runs on all 64 threads;
//   * the eight pair sums of each 64-key block, and the block sums, are independent of each other and are
//     computed in parallel; only the running total across blocks, sequential in the reference, stays on one
//     thread per query;
//   * the E4M3-published weights overwrite the exponentials in place once the sums have read them.
//
// Every value comes from the same functions on the same operands in the same order, so the output is
// bit-identical to `vit_attend`; NR_REFERENCE_KERNELS=1 selects that one (tools/compare_kernels.sh).
//
// Concatenated after numerics.wgsl and vit.wgsl, whose bindings, `params` and `normalized_at` it uses.

const QUERIES = 2u;

var<workgroup> pair_scores : array<array<f32, PADDED_TOKENS>, QUERIES>;
var<workgroup> pair_queries : array<array<f32, 32>, QUERIES>;
var<workgroup> pair_sums : array<array<f32, PADDED_TOKENS / 8u>, QUERIES>;
var<workgroup> block_sums : array<array<f32, PADDED_TOKENS / 64u>, QUERIES>;
var<workgroup> pair_reciprocals : array<f32, QUERIES>;

/// vit.wgsl's softmax_pair, over one query's scores.
fn query_softmax_pair(query: u32, base: u32, pair: u32, parity: u32) -> f32 {
  let key = base + pair * 2u + parity;
  let a = round_f16(pair_scores[query][key] + pair_scores[query][key + 8u]);
  let b = round_f16(pair_scores[query][key + 16u] + pair_scores[query][key + 24u]);
  let c = round_f16(pair_scores[query][key + 32u] + pair_scores[query][key + 40u]);
  let d = round_f16(pair_scores[query][key + 48u] + pair_scores[query][key + 56u]);
  return round_f16(round_f16(round_f16(a + b) + c) + d);
}

@compute @workgroup_size(64)
fn vit_attend_parallel(@builtin(workgroup_id) group : vec3<u32>, @builtin(local_invocation_index) thread : u32) {
  let head = group.x;
  let first_token = (group.y + group.z * 65535u) * QUERIES;
  if (first_token >= params.tokens) { return; }
  let stride3 = params.channels * 3u;
  let head_base = head * 96u;
  let blocks = PADDED_TOKENS / 64u;

  {
    let query = thread / 32u;
    let token = first_token + query;
    pair_queries[query][thread % 32u] = select(0.0, normalized_at(token * stride3 + head_base + thread % 32u),
                                               token < params.tokens);
  }
  workgroupBarrier();

  var a : array<f32, 16>;
  var b : array<f32, 16>;
  for (var item = thread; item < QUERIES * PADDED_TOKENS; item = item + 64u) {
    let query = item / PADDED_TOKENS;
    let key = item % PADDED_TOKENS;
    var score = 0.0;
    for (var half_step = 0u; half_step < 2u; half_step = half_step + 1u) {
      let c0 = half_step * 16u;
      for (var i = 0u; i < 16u; i = i + 1u) {
        a[i] = pair_queries[query][c0 + i];
        b[i] = normalized_at(key * stride3 + head_base + 32u + c0 + i);
      }
      score = ada_fp8_fdpa16(a, b, score);
    }
    pair_scores[query][key] = vit_exp_weight(score);
  }
  workgroupBarrier();

  // softmax64's eight pair sums per block: (block, pair, parity) -> pair_sums[block * 8 + parity * 4 + pair].
  for (var item = thread; item < QUERIES * blocks * 8u; item = item + 64u) {
    let query = item / (blocks * 8u);
    let i = item % (blocks * 8u);
    pair_sums[query][i] = query_softmax_pair(query, (i / 8u) * 64u, i % 4u, (i / 4u) % 2u);
  }
  workgroupBarrier();
  // softmax64 itself, in its own order, one (query, block) per thread.
  for (var item = thread; item < QUERIES * blocks; item = item + 64u) {
    let query = item / blocks;
    let even_base = (item % blocks) * 8u;
    let odd_base = even_base + 4u;
    let e0 = round_f16(pair_sums[query][even_base] + pair_sums[query][even_base + 1u]);
    let e1 = round_f16(e0 + pair_sums[query][even_base + 2u]);
    let even = round_f16(e1 + pair_sums[query][even_base + 3u]);
    let o0 = round_f16(pair_sums[query][odd_base] + pair_sums[query][odd_base + 1u]);
    let o1 = round_f16(o0 + pair_sums[query][odd_base + 2u]);
    let odd = round_f16(o1 + pair_sums[query][odd_base + 3u]);
    block_sums[query][item % blocks] = round_f16(even + odd);
  }
  workgroupBarrier();

  if (thread < QUERIES) {
    var total = 0.0;
    for (var block = 0u; block < blocks; block = block + 1u) {
      total = round_f16(total + block_sums[thread][block]);
    }
    let padding = PADDED_TOKENS - params.tokens;
    if (padding > 0u) {
      let correction = round_f16(vit_exp_weight(0.0) * f32(padding));
      total = round_f16(total - correction);
    }
    pair_reciprocals[thread] = round_f16(1.0 / total);
  }
  // The weights are published unnormalized, in place: nothing reads the exponentials after the sums.
  for (var item = thread; item < QUERIES * PADDED_TOKENS; item = item + 64u) {
    let query = item / PADDED_TOKENS;
    let key = item % PADDED_TOKENS;
    pair_scores[query][key] = decode_e4m3(encode_e4m3(f16_bits(pair_scores[query][key])));
  }
  workgroupBarrier();

  // Every thread: one component of one query, walking every key in k32 steps.
  let query = thread / 32u;
  let component = thread % 32u;
  var value = 0.0;
  for (var kb = 0u; kb < PADDED_TOKENS; kb = kb + 32u) {
    for (var half_step = 0u; half_step < 2u; half_step = half_step + 1u) {
      let k0 = kb + half_step * 16u;
      for (var i = 0u; i < 16u; i = i + 1u) {
        a[i] = pair_scores[query][k0 + i];
        b[i] = normalized_at((k0 + i) * stride3 + head_base + 64u + component);
      }
      value = ada_fp8_fdpa16(a, b, value);
    }
  }
  workgroupBarrier();   // every thread has read the weights; the first row is reused for the results
  pair_queries[query][component] = round_f16(value * pair_reciprocals[query]);
  workgroupBarrier();

  if (thread < QUERIES * 8u) {
    let token = first_token + thread / 8u;
    let quad = thread % 8u;
    if (token < params.tokens) {
      var word = 0u;
      for (var i = 0u; i < 4u; i = i + 1u) {
        word = word | (encode_e4m3(f16_bits(pair_queries[thread / 8u][quad * 4u + i])) << (i * 8u));
      }
      attended[((token * params.channels + head * 32u) >> 2u) + quad] = word;
    }
  }
}
