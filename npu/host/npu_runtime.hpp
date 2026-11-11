// C++ wrapper over liblfm3_npu.so, loaded at run time with dlopen.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "lfm3_npu.h"

namespace lfm3::npu {

class Runtime {
 public:
  // |library| is the path of liblfm3_npu.so; empty uses LFM3_NPU_LIBRARY or
  // the dynamic loader's search path.
  explicit Runtime(std::string library = {}, std::string device = "xdna://0?columns=8");
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  int Compile(std::string_view source, std::string_view name, std::string_view root);
  int Alloc(std::uint64_t bytes);
  void Free(int buffer);
  void Write(int buffer, std::uint64_t offset, const void* data, std::uint64_t bytes);
  void Read(int buffer, std::uint64_t offset, void* data, std::uint64_t bytes);
  void Dispatch(int kernel, std::span<const int> buffers);
  // Queues behind earlier dispatches; Finish, Read and Write wait.
  void Enqueue(int kernel, std::span<const int> buffers);
  void Finish();

  template <typename T>
  int Upload(std::span<const T> values) {
    const int id = Alloc(values.size_bytes());
    Write(id, 0, values.data(), values.size_bytes());
    return id;
  }

 private:
  void Check(int code, const char* what) const;
  void* handle_ = nullptr;
  lfm3_npu* npu_ = nullptr;
  struct Api;
  Api* api_ = nullptr;
};

}  // namespace lfm3::npu
