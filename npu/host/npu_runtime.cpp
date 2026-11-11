#include "npu_runtime.hpp"

#include <dlfcn.h>

#include <cstdlib>

namespace lfm3::npu {

struct Runtime::Api {
  int (*abi_version)();
  int (*open)(const char*, lfm3_npu**);
  void (*close)(lfm3_npu*);
  const char* (*error)(const lfm3_npu*);
  int (*compile)(lfm3_npu*, const char*, size_t, const char*, const char*, int*);
  int (*alloc)(lfm3_npu*, uint64_t, int*);
  int (*free)(lfm3_npu*, int);
  int (*write)(lfm3_npu*, int, uint64_t, const void*, uint64_t);
  int (*read)(lfm3_npu*, int, uint64_t, void*, uint64_t);
  int (*dispatch)(lfm3_npu*, int, const int*, int);
};

namespace {
template <typename F>
void Bind(void* handle, const char* name, F& out) {
  out = reinterpret_cast<F>(dlsym(handle, name));
  if (!out) throw std::runtime_error(std::string("liblfm3_npu: missing ") + name);
}
}  // namespace

Runtime::Runtime(std::string library, std::string device) {
  if (library.empty()) {
    const char* env = std::getenv("LFM3_NPU_LIBRARY");
    library = env ? env : "liblfm3_npu.so";
  }
  handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) throw std::runtime_error(std::string("dlopen: ") + dlerror());
  api_ = new Api();
  Bind(handle_, "lfm3_npu_abi_version", api_->abi_version);
  Bind(handle_, "lfm3_npu_open", api_->open);
  Bind(handle_, "lfm3_npu_close", api_->close);
  Bind(handle_, "lfm3_npu_error", api_->error);
  Bind(handle_, "lfm3_npu_compile", api_->compile);
  Bind(handle_, "lfm3_npu_alloc", api_->alloc);
  Bind(handle_, "lfm3_npu_free", api_->free);
  Bind(handle_, "lfm3_npu_write", api_->write);
  Bind(handle_, "lfm3_npu_read", api_->read);
  Bind(handle_, "lfm3_npu_dispatch", api_->dispatch);
  if (api_->abi_version() != LFM3_NPU_ABI_VERSION) {
    throw std::runtime_error("liblfm3_npu: ABI version mismatch");
  }
  Check(api_->open(device.c_str(), &npu_), "open");
}

Runtime::~Runtime() {
  if (api_ && npu_) api_->close(npu_);
  delete api_;
  // The HRX runtime keeps process-wide state; the library stays loaded.
}

void Runtime::Check(int code, const char* what) const {
  if (code != 0) {
    throw std::runtime_error(std::string("NPU ") + what + ": " +
                             (npu_ ? api_->error(npu_) : "open failed"));
  }
}

int Runtime::Compile(std::string_view source, std::string_view name, std::string_view root) {
  const std::string n(name), r(root);
  int id = -1;
  Check(api_->compile(npu_, source.data(), source.size(), n.c_str(), r.c_str(), &id), "compile");
  return id;
}

int Runtime::Alloc(std::uint64_t bytes) {
  int id = -1;
  Check(api_->alloc(npu_, bytes, &id), "alloc");
  return id;
}

void Runtime::Free(int buffer) { api_->free(npu_, buffer); }

void Runtime::Write(int buffer, std::uint64_t offset, const void* data, std::uint64_t bytes) {
  Check(api_->write(npu_, buffer, offset, data, bytes), "write");
}

void Runtime::Read(int buffer, std::uint64_t offset, void* data, std::uint64_t bytes) {
  Check(api_->read(npu_, buffer, offset, data, bytes), "read");
}

void Runtime::Dispatch(int kernel, std::span<const int> buffers) {
  Check(api_->dispatch(npu_, kernel, buffers.data(), static_cast<int>(buffers.size())), "dispatch");
}

}  // namespace lfm3::npu
