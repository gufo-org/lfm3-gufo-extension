// Checks liblfm3_xrt's embedding-row inputs against token ids: a prompt whose
// second half is given as embedding rows (copied from the bundle's table) must
// give the same logits, and so must a decode step fed a row.
// Usage: mixed_check <bundle> <ids.txt> <count>
#include "../../flm_lfm3_shim.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::ifstream f(argv[2]);
  std::vector<int32_t> ids;
  for (int v; f >> v;) ids.push_back(v);
  const int n = std::atoi(argv[3]);
  ids.resize(n + 1);
  flm_lfm3 *m = flm_lfm3_create(argv[1], "", 4096);
  if (!m) return std::printf("create: %s\n", flm_lfm3_last_error()), 1;
  const int V = flm_lfm3_vocab(m), H = 2048;
  // embedding rows straight from the bundle's decode state (f32 table)
  std::ifstream meta(std::string(argv[1]) + "/bundle.txt");
  long emb = 0, emb_row = 0;
  for (std::string k; meta >> k;) {
    if (k == "d.emb") meta >> emb;
    else if (k == "d.emb_row") meta >> emb_row;
    else meta.ignore(1 << 20, '\n');
  }
  FILE *s = std::fopen((std::string(argv[1]) + "/decode_state.bin").c_str(), "rb");
  std::vector<float> rows(size_t(n + 1) * H);
  for (int i = 0; i <= n; ++i) {
    std::fseek(s, emb + long(ids[i]) * emb_row, SEEK_SET);
    if (std::fread(&rows[size_t(i) * H], 4, H, s) != size_t(H)) return 1;
  }
  std::fclose(s);
  std::vector<float> a(V), b(V), c(V), d(V);
  flm_lfm3_clear(m);
  if (flm_lfm3_prefill(m, ids.data(), n, a.data())) return std::printf("prefill: %s\n", flm_lfm3_last_error()), 1;
  if (flm_lfm3_decode(m, ids[n], c.data())) return 1;
  std::vector<int32_t> mixed(ids.begin(), ids.begin() + n);
  for (int i = n / 2; i < n; ++i) mixed[i] = -(i + 1);
  flm_lfm3_clear(m);
  if (flm_lfm3_prefill_mixed(m, mixed.data(), rows.data(), n, b.data())) return std::printf("mixed: %s\n", flm_lfm3_last_error()), 1;
  if (flm_lfm3_decode_row(m, &rows[size_t(n) * H], d.data())) return std::printf("row: %s\n", flm_lfm3_last_error()), 1;
  double e1 = 0, e2 = 0;
  for (int i = 0; i < V; ++i) e1 = std::fmax(e1, std::fabs(a[i] - b[i])), e2 = std::fmax(e2, std::fabs(c[i] - d[i]));
  std::printf("max |logit diff|: prefill ids vs rows %.3g, decode id vs row %.3g\n", e1, e2);
  flm_lfm3_destroy(m);
  return e1 > 0 || e2 > 0;
}
