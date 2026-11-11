// Compiles a Loom file's root on the NPU and prints the result (code size
// diagnostics included). Usage: npu_compile_check <file.loom> <@root>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "npu_runtime.hpp"

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::stringstream text;
  text << in.rdbuf();
  lfm3::npu::Runtime npu;
  try {
    npu.Compile(text.str(), argv[1], argv[2]);
    std::printf("ok\n");
  } catch (const std::exception& e) {
    std::string m = e.what();
    const auto at = m.find("requires ");
    std::printf("%s\n", at == std::string::npos ? m.substr(0, 400).c_str() : m.substr(at, 40).c_str());
    return 1;
  }
}
