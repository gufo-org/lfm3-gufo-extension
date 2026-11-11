#include "fuse.hpp"

#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace gufo::models::lfm3::npu {

std::string FuseProgram(const std::vector<FuseStage>& stages, int bindings, const std::string& name) {
  static const std::regex symbol(R"(@([A-Za-z0-9_]+)[\(\[])");
  static const std::string kPublic = "pipeline.def<kernel> public retain target(@npu) ";
  std::set<std::string> seen;
  std::string out = "aie2p.target<array> @npu\n";
  for (const auto& stage : stages) {
    std::istringstream in(stage.program);
    std::string line, item, key;
    auto flush = [&] {
      if (!item.empty() && seen.insert(key).second) out += item;
      item.clear();
      key.clear();
    };
    while (std::getline(in, line)) {
      if (item.empty()) {
        if (line.rfind("func.def", 0) != 0 && line.rfind("pipeline.def", 0) != 0) continue;
        std::smatch m;
        if (!std::regex_search(line, m, symbol)) throw std::logic_error("fuse: unnamed definition: " + line);
        key = m[1];
        if (line.rfind(kPublic, 0) == 0) line = "pipeline.def<kernel> " + line.substr(kPublic.size());
      }
      item += line + "\n";
      if (line == "}") flush();
    }
    if (!item.empty()) throw std::logic_error("fuse: unterminated definition in " + stage.root);
  }
  std::string args, types;
  for (int b = 0; b < bindings; ++b) args += (b ? ", %b" : "%b") + std::to_string(b) + ": buffer";
  out += "\npipeline.def<kernel> public retain target(@npu) @" + name + "() run(" + args + ") {\n  pipeline.serial {\n";
  for (const auto& stage : stages) {
    std::string a, t;
    for (std::size_t i = 0; i < stage.args.size(); ++i) {
      if (stage.args[i] < 0 || stage.args[i] >= bindings) throw std::logic_error("fuse: bad binding");
      a += (i ? ", %b" : "%b") + std::to_string(stage.args[i]);
      t += i ? ", buffer" : "buffer";
    }
    out += "    pipeline.compose " + stage.root + "(" + a + ") : (" + t + ")\n";
  }
  out += "  }\n}\n";
  return out;
}

}  // namespace gufo::models::lfm3::npu
