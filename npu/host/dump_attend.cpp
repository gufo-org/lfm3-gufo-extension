#include <cstdio>
#include "../attention.hpp"
using namespace gufo::models::lfm3::npu;
int main() {
  AttentionShape s{16, 4, 128, 392, true, 128, 4, 64, 9, false};
  std::fputs(AttendProgram(s, "attend").c_str(), stdout);
}
