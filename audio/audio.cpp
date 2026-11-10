#include <algorithm>
#include <cmath>
#include <exception>
#include <gufo/ext/wav.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <thread>

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
// Hann-windowed sinc with the cutoff below the lower Nyquist rate. The
// filter of an output depends only on its phase, the fractional part of its
// input time, so rational rates tabulate it once per phase (two for 24 kHz,
// one for 48 kHz, 160 for 44.1 kHz) instead of evaluating it per tap.
Resampler::Resampler(std::uint32_t from, std::uint32_t to)
    : from_(from),
      to_(to),
      step_(from / std::gcd(from, to)),
      phases_(to / std::gcd(from, to)),
      cutoff_(std::min(1.0, double(to) / from) * 0.95),
      half_(24 / cutoff_),
      same_(from == to) {
  if (same_)
    return;
  first_.resize(phases_);
  count_.resize(phases_);
  for (std::uint64_t k = 0; k < phases_; ++k) {
    const double frac = double(k) / double(phases_);
    first_[k] = std::int32_t(std::ceil(frac - half_));
    count_[k] = std::int32_t(std::floor(frac + half_)) - first_[k] + 1;
    taps_ = std::max(taps_, std::size_t(count_[k]));
  }
  // Coprime rates have up to 16000 phases; past 8 MB of weights each
  // output computes its own.
  if (phases_ * taps_ > (std::size_t(1) << 20))
    return;
  weights_.assign(phases_ * taps_, 0.0);
  for (std::uint64_t k = 0; k < phases_; ++k)
    for (std::int32_t i = 0; i < count_[k]; ++i)
      weights_[k * taps_ + i] =
          Weight(double(k) / double(phases_) - (first_[k] + i));
}
double Resampler::Weight(double d) const {
  const double a = std::numbers::pi * cutoff_ * d;
  const double sinc = std::abs(a) < 1e-9 ? 1.0 : std::sin(a) / a;
  const double window = 0.5 + 0.5 * std::cos(std::numbers::pi * d / half_);
  return cutoff_ * sinc * window;
}
std::vector<float> Resampler::Push(std::span<const float> x, bool finish) {
  if (same_)
    return {x.begin(), x.end()};
  // Each output waits until its window has arrived.
  history_.insert(history_.end(), x.begin(), x.end());
  received_ += x.size();
  const auto total = std::int64_t(received_);
  const auto end = finish ? (received_ * to_ + from_ - 1) / from_
                          : std::numeric_limits<std::uint64_t>::max();
  // First input of output i's window.
  const auto window_start = [&](std::uint64_t i, std::uint64_t* phase) {
    const auto position = i * step_;
    *phase = position % phases_;
    return std::int64_t(position / phases_) + first_[*phase];
  };
  // Output i, whose window starts at first; scratch holds taps_ weights
  // when they are not tabulated.
  const auto output = [&](std::uint64_t k, std::int64_t first,
                          double* scratch) {
    const auto lo = std::max<std::int64_t>(first, 0);
    const auto hi = std::min<std::int64_t>(first + count_[k] - 1, total - 1);
    const double* w;
    if (weights_.empty()) {
      const double frac = double(k) / double(phases_);
      for (std::int32_t i = 0; i < count_[k]; ++i)
        scratch[i] = Weight(frac - (first_[k] + i));
      w = scratch;
    } else
      w = weights_.data() + k * taps_;
    w += lo - first;
    const float* in = history_.data() + (lo - base_);
    const auto n = std::size_t(std::max<std::int64_t>(hi - lo + 1, 0));
    // Independent partial sums let the taps vectorize.
    double part[8] = {};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
      for (int lane = 0; lane < 8; ++lane)
        part[lane] += double(in[i + lane]) * w[i + lane];
    for (; i < n; ++i)
      part[i % 8] += double(in[i]) * w[i];
    double sum = 0;
    for (double p : part)
      sum += p;
    return float(sum);
  };
  std::vector<float> y;
  std::vector<double> scratch(weights_.empty() ? taps_ : 0);
  if (!finish) {
    for (;; ++next_) {
      std::uint64_t k;
      const auto first = window_start(next_, &k);
      if (first + count_[k] - 1 >= total)
        break;
      y.push_back(output(k, first, scratch.data()));
    }
  } else if (end > next_) {
    // The remaining outputs are independent: a long clip spreads them over
    // threads.
    y.resize(end - next_);
    const auto run = [&](std::uint64_t begin, std::uint64_t stop) {
      std::vector<double> own(scratch.size());
      for (auto i = begin; i < stop; ++i) {
        std::uint64_t k;
        const auto first = window_start(i, &k);
        y[i - next_] = output(k, first, own.data());
      }
    };
    const auto count = end - next_;
    const auto threads = std::uint64_t(
        count < 16000
            ? 1
            : std::clamp(std::thread::hardware_concurrency(), 1U, 8U));
    std::vector<std::thread> workers;
    for (std::uint64_t t = 1; t < threads; ++t)
      workers.emplace_back(run, next_ + count * t / threads,
                           next_ + count * (t + 1) / threads);
    run(next_, next_ + count / threads);
    for (auto& worker : workers)
      worker.join();
    next_ = end;
  }
  // Drop input the next output's window no longer reaches.
  std::uint64_t k;
  const auto keep =
      std::clamp<std::int64_t>(window_start(next_, &k), base_, total);
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
