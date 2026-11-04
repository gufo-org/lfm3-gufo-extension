#ifndef GUFO_MODELS_LFM3_CONTINUATION_HOOKS_HPP_
#define GUFO_MODELS_LFM3_CONTINUATION_HOOKS_HPP_
#include <cstdint>
namespace gufo::models::lfm3 {
// Installed only in sessions owned by the continuation adapter. Every callback
// runs on the host before the session changes state or submits device work,
// and may throw to refuse it.
class ContinuationHooks {
public:
  virtual ~ContinuationHooks() = default;
  // Rows [first, end) are about to be written.
  virtual void BeforeWrite(std::uint32_t first, std::uint32_t end) = 0;
  virtual void BeforeReset() = 0;
  virtual void BeforeRestore() = 0;
};
}  // namespace gufo::models::lfm3
#endif
