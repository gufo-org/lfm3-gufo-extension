#pragma once
// d1's grouped full attention on the host (npu::Backbone).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/experimental/npu.hpp>
#include <iterator>
#include <map>
#include <span>
#include <vector>

namespace gufo::models::lfm3::npu {

// d1 full attention with question groups: a row of group -1 sees every row,
// another row the -1 rows and its own group. Each set of queries (the -1
// rows, then each group) runs host::Attention over its keys gathered in
// position order. q rows at q_stride, k/v rows at kv_stride, out at
// out_stride.
inline void GroupedAttention(const float* q, std::size_t q_stride,
                             const float* k, const float* v,
                             std::size_t kv_stride, float* out,
                             std::size_t out_stride,
                             std::span<const std::int32_t> groups,
                             std::size_t heads, std::size_t kv_heads,
                             std::size_t dim, float scale) {
  const std::size_t n = groups.size(), Q = heads * dim, KV = kv_heads * dim;
  std::map<std::int32_t, std::vector<std::size_t>> members;
  for (std::size_t i = 0; i < n; ++i)
    members[groups[i] < 0 ? -1 : groups[i]].push_back(i);
  const auto state = members.find(-1);
  std::vector<float> qs, ks, vs, os;
  for (const auto& [group, rows] : members) {
    std::vector<std::size_t> keys;
    if (group < 0) {
      keys.resize(n);
      for (std::size_t t = 0; t < n; ++t)
        keys[t] = t;
    } else {
      if (state != members.end())
        std::merge(state->second.begin(), state->second.end(), rows.begin(),
                   rows.end(), std::back_inserter(keys));
      else
        keys = rows;
    }
    qs.resize(rows.size() * Q);
    os.resize(rows.size() * Q);
    ks.resize(keys.size() * KV);
    vs.resize(keys.size() * KV);
    for (std::size_t i = 0; i < rows.size(); ++i)
      std::copy_n(q + rows[i] * q_stride, Q, qs.data() + i * Q);
    for (std::size_t t = 0; t < keys.size(); ++t) {
      std::copy_n(k + keys[t] * kv_stride, KV, ks.data() + t * KV);
      std::copy_n(v + keys[t] * kv_stride, KV, vs.data() + t * KV);
    }
    const std::size_t count = keys.size();
    gufo::npu::host::AttentionArgs args{
        qs.data(), Q,           ks.data(), KV,       vs.data(), KV,  os.data(),
        Q,         rows.size(), heads,     kv_heads, dim,       dim, scale};
    gufo::npu::host::Attention(
        args, [count](std::size_t, std::size_t* b, std::size_t* e) {
          *b = 0;
          *e = count;
        });
    for (std::size_t i = 0; i < rows.size(); ++i)
      std::copy_n(os.data() + i * Q, Q, out + rows[i] * out_stride);
  }
}

}  // namespace gufo::models::lfm3::npu
