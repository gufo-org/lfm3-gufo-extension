#include <algorithm>
#include <cmath>
#include <exception>
#include <gufo/ext/wav.hpp>
#include <limits>
#include <numbers>

#include "model.hpp"
namespace gufo::models::lfm3_audio {
namespace {
namespace core = gufo::core;

// Accepts PCM16/24/32 and float32 up to 8 channels and 192 kHz; 300 seconds
// matches the d1-Omni input contract, and the ASR runner windows longer
// input regardless. The shared decoder rejects non-finite float samples
// and clamps the rest to [-1, 1].
constexpr core::WavLimits kWavLimits{
    .max_channels = 8,
    .max_sample_rate = 192000,
    .max_seconds = 300,
    .max_pcm_bits = 32,
};
constexpr core::WavMessages kWavMessages{
    .not_riff = "audio must be a RIFF/WAVE file",
    .truncated = "truncated WAV chunk",
    .invalid_fmt = "WAV fmt chunk is invalid",
    .unsupported_format = "WAV format is unsupported",
    .unsupported_layout = "WAV must be PCM16/24/32 or float32 interleaved",
    .too_long = "audio must contain at most 300 seconds",
    .non_finite = "WAV contains non-finite samples",
};

void Write32(std::vector<std::byte>& out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(std::byte((v >> (8 * i)) & 255));
}
void Write16(std::vector<std::byte>& out, std::uint16_t v) {
  out.push_back(std::byte(v & 255));
  out.push_back(std::byte(v >> 8));
}
}  // namespace
bool DecodeWav(std::span<const std::byte> wav, std::vector<float>* samples,
               std::uint32_t* rate, std::string* error) {
  try {
    const core::WavAudio decoded =
        core::DecodeWav(wav, kWavLimits, kWavMessages);
    const std::size_t channels = decoded.channels;
    samples->assign(decoded.samples.size() / channels, 0.0F);
    for (std::size_t frame = 0; frame < samples->size(); ++frame) {
      float sum = 0.0F;
      for (std::size_t c = 0; c < channels; ++c)
        sum += decoded.samples[frame * channels + c];
      (*samples)[frame] = sum / float(channels);
    }
    *rate = decoded.sample_rate;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
std::vector<std::byte> EncodeWav(std::span<const float> samples,
                                 std::uint32_t rate) {
  std::vector<std::byte> out;
  out.reserve(44 + samples.size() * 2);
  const auto data_bytes = std::uint32_t(samples.size() * 2);
  for (char c : std::string_view("RIFF"))
    out.push_back(std::byte(c));
  Write32(out, 36 + data_bytes);
  for (char c : std::string_view("WAVEfmt "))
    out.push_back(std::byte(c));
  Write32(out, 16);
  Write16(out, 1);
  Write16(out, 1);
  Write32(out, rate);
  Write32(out, rate * 2);
  Write16(out, 2);
  Write16(out, 16);
  for (char c : std::string_view("data"))
    out.push_back(std::byte(c));
  Write32(out, data_bytes);
  for (float v : samples) {
    const float clipped = std::clamp(v, -1.0F, 1.0F);
    Write16(out, std::uint16_t(std::int16_t(std::lround(clipped * 32767.0F))));
  }
  return out;
}
Resampler::Resampler(std::uint32_t from, std::uint32_t to)
    : ratio_(double(to) / from),
      cutoff_(std::min(1.0, ratio_) * 0.95),
      half_(24 / cutoff_),
      same_(from == to) {}
std::vector<float> Resampler::Push(std::span<const float> x, bool finish) {
  if (same_)
    return {x.begin(), x.end()};
  // Hann-windowed sinc with the cutoff below the lower Nyquist rate. Output
  // i sits at input time i / ratio and waits until its window has arrived.
  history_.insert(history_.end(), x.begin(), x.end());
  received_ += x.size();
  const auto total = std::int64_t(received_);
  const auto end = finish ? std::uint64_t(std::ceil(received_ * ratio_))
                          : std::numeric_limits<std::uint64_t>::max();
  std::vector<float> y;
  for (; next_ < end; ++next_) {
    const double t = next_ / ratio_;
    const auto first = std::int64_t(std::ceil(t - half_));
    const auto last = std::int64_t(std::floor(t + half_));
    if (!finish && last >= total)
      break;
    double sum = 0;
    for (auto j = std::max<std::int64_t>(first, 0);
         j <= std::min<std::int64_t>(last, total - 1); ++j) {
      const double d = t - double(j);
      const double a = std::numbers::pi * cutoff_ * d;
      const double sinc = std::abs(a) < 1e-9 ? 1.0 : std::sin(a) / a;
      const double window = 0.5 + 0.5 * std::cos(std::numbers::pi * d / half_);
      sum += history_[std::size_t(j - base_)] * cutoff_ * sinc * window;
    }
    y.push_back(float(sum));
  }
  // Drop input the next output's window no longer reaches.
  const auto keep = std::clamp<std::int64_t>(
      std::int64_t(std::ceil(next_ / ratio_ - half_)), base_, total);
  history_.erase(history_.begin(), history_.begin() + (keep - base_));
  base_ = keep;
  return y;
}
std::vector<float> Resample(std::span<const float> x, std::uint32_t from,
                            std::uint32_t to) {
  if (x.empty())
    return {};
  return Resampler(from, to).Push(x, true);
}
}  // namespace gufo::models::lfm3_audio
