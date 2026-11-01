#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <future>
#include <gufo/ext/crypto.hpp>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/fs.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include <gufo/ext/platform/os_info.hpp>
#include <gufo/ext/sampling.hpp>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace gufo::quant {

float Fp16ToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      float zero = 0.0f;
      std::memcpy(&zero, &sign, sizeof(float));
      return zero;
    }
    const float f =
        (static_cast<float>(mant) / 1024.0f) * std::ldexp(1.0f, -14);
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    const std::uint32_t val = sign | 0x7F800000U | (mant << 13);
    float f = 0.0f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
  }

  const std::uint32_t val = sign | ((exp + 112) << 23) | (mant << 13);
  float f = 0.0f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

std::size_t QuantizedRowBytes(core::GgmlType type,
                              std::size_t elements) noexcept {
  const std::size_t block_qk = QuantizedBlockElements(type);
  std::size_t block_bytes = 0;
  switch (type) {
    case core::GgmlType::kQ4_0:
      block_bytes = 18;
      break;
    case core::GgmlType::kQ4_1:
      block_bytes = 20;
      break;
    case core::GgmlType::kQ5_0:
      block_bytes = 22;
      break;
    case core::GgmlType::kQ5_1:
      block_bytes = 24;
      break;
    case core::GgmlType::kQ8_1:
      block_bytes = 36;
      break;
    case core::GgmlType::kQ2_K:
      block_bytes = 84;
      break;
    case core::GgmlType::kIQ2_XXS:
      block_bytes = 66;
      break;
    case core::GgmlType::kIQ2_XS:
      block_bytes = 74;
      break;
    case core::GgmlType::kIQ2_S:
      block_bytes = 82;
      break;
    case core::GgmlType::kIQ3_XXS:
      block_bytes = 98;
      break;
    case core::GgmlType::kIQ1_S:
      block_bytes = 50;
      break;
    case core::GgmlType::kIQ1_M:
      block_bytes = 56;
      break;
    case core::GgmlType::kQ3_K:
      block_bytes = sizeof(block_q3_K);
      break;
    case core::GgmlType::kQ4_K:
      block_bytes = sizeof(block_q4_K);
      break;
    case core::GgmlType::kQ5_K:
      block_bytes = sizeof(block_q5_K);
      break;
    case core::GgmlType::kQ6_K:
      block_bytes = sizeof(block_q6_K);
      break;
    case core::GgmlType::kQ8_K:
      block_bytes = sizeof(block_q8_K);
      break;
    case core::GgmlType::kQ8_0:
      block_bytes = sizeof(block_q8_0);
      break;
    case core::GgmlType::kIQ4_NL:
      block_bytes = sizeof(block_iq4_nl);
      break;
    case core::GgmlType::kIQ4_XS:
      block_bytes = sizeof(block_iq4_xs);
      break;
    case core::GgmlType::kIQ3_S:
      block_bytes = sizeof(block_iq3_s);
      break;
    default:
      return 0;
  }
  if ((elements % block_qk) != 0) {
    return 0;
  }
  const std::size_t blocks = elements / block_qk;
  if (blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
    return 0;
  }
  return blocks * block_bytes;
}

std::size_t EncodedSizeBytes(core::GgmlType type,
                             std::size_t elements) noexcept {
  switch (type) {
    case core::GgmlType::kI32:
      if (elements >
          std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        return 0;
      }
      return elements * sizeof(std::int32_t);
    case core::GgmlType::kF32:
      if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return 0;
      }
      return elements * sizeof(float);
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      if (elements >
          std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        return 0;
      }
      return elements * sizeof(std::uint16_t);
    default:
      return QuantizedRowBytes(type, elements);
  }
}

void DequantizeQ8_0(const void* src, float* dst, std::size_t k) {
  const auto* blocks = static_cast<const block_q8_0*>(src);
  const std::size_t nb = k / 32;

  for (std::size_t b = 0; b < nb; ++b) {
    const float d = Fp16ToFloat(blocks[b].d);
    for (std::size_t i = 0; i < 32; ++i) {
      dst[(b * 32) + i] = d * static_cast<float>(blocks[b].qs[i]);
    }
  }
}

}  // namespace gufo::quant

namespace gufo::core {
namespace {

constexpr std::array<char, 4> kGgufMagic = {'G', 'G', 'U', 'F'};

template<typename T>
bool ReadPod(const std::uint8_t* data, std::size_t size, std::size_t& offset,
             T& out) {
  if (offset > size || sizeof(T) > size - offset) {
    return false;
  }
  std::memcpy(&out, data + offset, sizeof(T));
  offset += sizeof(T);
  return true;
}

bool ReadString(const std::uint8_t* data, std::size_t size, std::size_t& offset,
                std::string_view& out) {
  std::uint64_t len = 0;
  if (!ReadPod(data, size, offset, len)) {
    return false;
  }
  if (offset > size || len > size - offset) {
    return false;
  }
  out = std::string_view(reinterpret_cast<const char*>(data + offset), len);
  offset += len;
  return true;
}

}  // namespace

GgufReader::~GgufReader() = default;

GgufReader::GgufReader(GgufReader&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      source_path_(std::move(other.source_path_)),
      mapping_(std::move(other.mapping_)),
      version_(other.version_),
      alignment_(other.alignment_),
      metadata_(std::move(other.metadata_)),
      tensors_(std::move(other.tensors_)),
      tensor_index_(std::move(other.tensor_index_)),
      mapped_regions_(std::move(other.mapped_regions_)),
      shards_(std::move(other.shards_)) {
  other.data_ = nullptr;
  other.size_ = 0;
}

GgufReader& GgufReader::operator=(GgufReader&& other) noexcept {
  if (this != &other) {
    mapping_.Close();
    shards_.clear();
    mapped_regions_.clear();

    data_ = other.data_;
    size_ = other.size_;
    source_path_ = std::move(other.source_path_);
    mapping_ = std::move(other.mapping_);
    version_ = other.version_;
    alignment_ = other.alignment_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    tensor_index_ = std::move(other.tensor_index_);
    mapped_regions_ = std::move(other.mapped_regions_);
    shards_ = std::move(other.shards_);

    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

std::unique_ptr<GgufReader> GgufReader::OpenFile(
    const std::filesystem::path& path, std::string* error_msg) {
  auto first = OpenSingleFile(path, error_msg);
  if (!first) {
    return nullptr;
  }

  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  if (first->FindMetadata("split.count") &&
      (!first->GetMetadataUint32("split.count") || split_count == 0 ||
       split_count > 65535)) {
    if (error_msg)
      *error_msg = "Invalid GGUF split count";
    return nullptr;
  }
  if (split_count <= 1) {
    return first;
  }
  return OpenSplitFileSet(path, std::move(first), error_msg);
}

std::unique_ptr<GgufReader> GgufReader::OpenSingleFile(
    const std::filesystem::path& path, std::string* error_msg) {
  std::string open_error;
  auto mapping = platform::ReadOnlyMappedFile::Open(path, &open_error);
  if (!mapping.valid()) {
    if (error_msg != nullptr) {
      *error_msg =
          "Failed to open or map file: " + path.string() + ": " + open_error;
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->data_ = mapping.data();
  reader->size_ = mapping.size();
  reader->source_path_ = path;
  reader->mapping_ = std::move(mapping);

  try {
    if (!reader->ParseHeaders(error_msg)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return nullptr;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return nullptr;
  }
  reader->mapped_regions_.push_back({reader->data_, reader->size_,
                                     reader->mapping_.PlatformHandle(),
                                     reader->source_path_});
  return reader;
}

std::unique_ptr<GgufReader> GgufReader::OpenSplitFileSet(
    const std::filesystem::path& path, std::unique_ptr<GgufReader> first,
    std::string* error_msg) {
  const auto split_count = first->GetMetadataUint32("split.count").value_or(1U);
  const auto split_no = first->GetMetadataUint32("split.no");
  const auto total_tensor_count =
      first->GetMetadataUint64("split.tensors.count");
  if (!split_no.has_value() || *split_no >= split_count ||
      !total_tensor_count.has_value()) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid split GGUF metadata in " + path.string();
    }
    return nullptr;
  }

  const std::string filename = path.filename().string();
  const std::regex split_pattern{R"(^(.*)-([0-9]+)-of-([0-9]+)\.gguf$)"};
  std::smatch match;
  if (!std::regex_match(filename, match, split_pattern)) {
    if (error_msg != nullptr) {
      *error_msg =
          "Split GGUF filename does not match "
          "<name>-00001-of-00002.gguf: " +
          filename;
    }
    return nullptr;
  }

  const std::string prefix = match[1].str();
  const std::size_t index_width = match[2].str().size();
  const std::size_t count_width = match[3].str().size();
  const auto build_path = [&](std::uint32_t index) {
    std::ostringstream name;
    name << prefix << '-' << std::setfill('0')
         << std::setw(static_cast<int>(index_width)) << (index + 1) << "-of-"
         << std::setw(static_cast<int>(count_width)) << split_count << ".gguf";
    return path.parent_path() / name.str();
  };

  std::vector<std::unique_ptr<GgufReader>> shards(split_count);
  shards[*split_no] = std::move(first);
  for (std::uint32_t i = 0; i < split_count; ++i) {
    if (shards[i]) {
      continue;
    }
    const auto shard_path = build_path(i);
    shards[i] = OpenSingleFile(shard_path, error_msg);
    if (!shards[i]) {
      if (error_msg != nullptr && error_msg->empty()) {
        *error_msg = "Failed to open split GGUF shard: " + shard_path.string();
      }
      return nullptr;
    }
  }

  const auto architecture =
      shards[0]->GetMetadataString("general.architecture");
  for (std::uint32_t i = 0; i < split_count; ++i) {
    const auto shard_no = shards[i]->GetMetadataUint32("split.no");
    const auto shard_count = shards[i]->GetMetadataUint32("split.count");
    const auto shard_architecture =
        shards[i]->GetMetadataString("general.architecture");
    if (!shard_no.has_value() || *shard_no != i || !shard_count.has_value() ||
        *shard_count != split_count ||
        shards[i]->GetVersion() != shards[0]->GetVersion() ||
        (shard_architecture.has_value() &&
         shard_architecture != architecture)) {
      if (error_msg != nullptr) {
        *error_msg =
            "Inconsistent split GGUF shard metadata: " + build_path(i).string();
      }
      return nullptr;
    }
  }

  auto combined = std::unique_ptr<GgufReader>(new GgufReader());
  combined->version_ = shards[0]->version_;
  combined->alignment_ = shards[0]->alignment_;
  combined->metadata_ = shards[0]->metadata_;
  std::size_t actual_count = 0;
  for (const auto& shard : shards) {
    if (shard->tensors_.size() >
            std::numeric_limits<std::size_t>::max() - actual_count ||
        shard->size_ >
            std::numeric_limits<std::size_t>::max() - combined->size_)
      return nullptr;
    actual_count += shard->tensors_.size();
    combined->size_ += shard->size_;
  }
  if (actual_count != *total_tensor_count) {
    if (error_msg)
      *error_msg = "Split GGUF tensor count mismatch";
    return nullptr;
  }
  combined->tensors_.reserve(actual_count);
  combined->tensor_index_.reserve(
      static_cast<std::size_t>(*total_tensor_count));
  combined->mapped_regions_.reserve(split_count);

  for (const auto& shard : shards) {
    combined->mapped_regions_.push_back({shard->data_, shard->size_,
                                         shard->mapping_.PlatformHandle(),
                                         shard->source_path_});
    for (const auto& tensor : shard->tensors_) {
      if (combined->tensor_index_.contains(tensor.name)) {
        if (error_msg != nullptr) {
          *error_msg = "Duplicate tensor across split GGUF shards: " +
                       std::string(tensor.name);
        }
        return nullptr;
      }
      combined->tensor_index_[tensor.name] = combined->tensors_.size();
      combined->tensors_.push_back(tensor);
    }
  }

  if (combined->tensors_.size() != *total_tensor_count) {
    if (error_msg != nullptr) {
      *error_msg = "Split GGUF tensor count mismatch: expected " +
                   std::to_string(*total_tensor_count) + ", found " +
                   std::to_string(combined->tensors_.size());
    }
    return nullptr;
  }

  combined->shards_ = std::move(shards);
  return combined;
}

std::unique_ptr<GgufReader> GgufReader::OpenMemory(const void* data,
                                                   std::size_t size,
                                                   std::string* error_msg) {
  if (data == nullptr || size < 24) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid memory buffer for GGUF";
    }
    return nullptr;
  }

  auto reader = std::unique_ptr<GgufReader>(new GgufReader());
  reader->data_ = static_cast<const std::uint8_t*>(data);
  reader->size_ = size;

  try {
    if (!reader->ParseHeaders(error_msg)) {
      if (error_msg && error_msg->empty())
        *error_msg = "Malformed or truncated GGUF";
      return nullptr;
    }
  } catch (const std::bad_alloc&) {
    if (error_msg)
      *error_msg = "Insufficient memory for GGUF descriptors";
    return nullptr;
  }
  reader->mapped_regions_.push_back(
      {reader->data_, reader->size_, platform::kInvalidFile, {}});
  return reader;
}

bool GgufReader::ParseHeaders(std::string* error_msg) {
  if (size_ < 24) {
    if (error_msg != nullptr) {
      *error_msg = "File too small for GGUF header";
    }
    return false;
  }

  if (std::memcmp(data_, kGgufMagic.data(), 4) != 0) {
    if (error_msg != nullptr) {
      *error_msg = "Invalid GGUF magic header";
    }
    return false;
  }

  std::size_t offset = 4;
  if (!ReadPod(data_, size_, offset, version_)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read GGUF version";
    }
    return false;
  }

  if (version_ != 2 && version_ != 3) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported GGUF version: " + std::to_string(version_);
    }
    return false;
  }

  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  if (!ReadPod(data_, size_, offset, tensor_count) ||
      !ReadPod(data_, size_, offset, metadata_count)) {
    if (error_msg != nullptr) {
      *error_msg = "Failed to read tensor/metadata counts";
    }
    return false;
  }

  std::size_t budget = 128 * 1024 * 1024;
  const auto charge = [&](std::uint64_t count, std::size_t item_bytes) {
    if (count > budget / item_bytes)
      return false;
    budget -= static_cast<std::size_t>(count) * item_bytes;
    return true;
  };
  if (metadata_count > (size_ - offset) / 13 ||
      tensor_count > (size_ - offset) / 32 ||
      !charge(metadata_count, sizeof(GgufMetadataValue) + 64) ||
      !charge(tensor_count, sizeof(GgufTensorInfo) + 128)) {
    if (error_msg)
      *error_msg = "GGUF descriptor counts exceed file or memory limits";
    return false;
  }
  metadata_.reserve(metadata_count);

  for (std::uint64_t i = 0; i < metadata_count; ++i) {
    std::string_view key;
    if (!ReadString(data_, size_, offset, key)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata key";
      }
      return false;
    }

    if (key.empty() || key.size() > 65535 || !charge(key.size(), 1)) {
      if (error_msg)
        *error_msg = "GGUF metadata key exceeds limits";
      return false;
    }
    std::uint32_t val_type_raw = 0;
    if (!ReadPod(data_, size_, offset, val_type_raw)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read metadata type for key " + std::string(key);
      }
      return false;
    }

    if (val_type_raw > static_cast<std::uint32_t>(GgufValueType::kFloat64))
      return false;
    const auto val_type = static_cast<GgufValueType>(val_type_raw);
    GgufMetadataValue meta_val;
    meta_val.type = val_type;

    switch (val_type) {
      case GgufValueType::kUint8: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt8: {
        std::int8_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint16: {
        std::uint16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt16: {
        std::int16_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kUint32: {
        std::uint32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::uint64_t>(v);
        break;
      }
      case GgufValueType::kInt32: {
        std::int32_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<std::int64_t>(v);
        break;
      }
      case GgufValueType::kFloat32: {
        float v = 0.0F;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = static_cast<double>(v);
        break;
      }
      case GgufValueType::kBool: {
        std::uint8_t v = 0;
        if (!ReadPod(data_, size_, offset, v) || v > 1) {
          return false;
        }
        meta_val.value = (v != 0);
        break;
      }
      case GgufValueType::kString: {
        std::string_view s;
        if (!ReadString(data_, size_, offset, s)) {
          return false;
        }
        if (!charge(s.size(), 1))
          return false;
        meta_val.value = s;
        break;
      }
      case GgufValueType::kUint64: {
        std::uint64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kInt64: {
        std::int64_t v = 0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kFloat64: {
        double v = 0.0;
        if (!ReadPod(data_, size_, offset, v)) {
          return false;
        }
        meta_val.value = v;
        break;
      }
      case GgufValueType::kArray: {
        std::uint32_t item_type_raw = 0;
        std::uint64_t array_len = 0;
        if (!ReadPod(data_, size_, offset, item_type_raw) ||
            !ReadPod(data_, size_, offset, array_len)) {
          return false;
        }
        if (item_type_raw >
                static_cast<std::uint32_t>(GgufValueType::kFloat64) ||
            item_type_raw == static_cast<std::uint32_t>(GgufValueType::kArray))
          return false;
        constexpr std::size_t item_bytes[] = {1, 1, 2, 2, 4, 4, 4,
                                              1, 8, 0, 8, 8, 8};
        if (offset > size_ ||
            array_len > (size_ - offset) / item_bytes[item_type_raw] ||
            !charge(array_len,
                    item_type_raw == 8 ? sizeof(std::string_view) : 8))
          return false;
        const auto item_type = static_cast<GgufValueType>(item_type_raw);
        if (item_type == GgufValueType::kString) {
          std::vector<std::string_view> str_arr;
          str_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::string_view s;
            if (!ReadString(data_, size_, offset, s)) {
              return false;
            }
            if (!charge(s.size(), 1))
              return false;
            str_arr.push_back(s);
          }
          meta_val.value = std::move(str_arr);
        } else if (item_type == GgufValueType::kFloat32 ||
                   item_type == GgufValueType::kFloat64) {
          const auto bytes = item_type == GgufValueType::kFloat32 ? 4U : 8U;
          if (offset > size_ || array_len > (size_ - offset) / bytes)
            return false;
          std::vector<double> values;
          values.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            double value = 0;
            if (item_type == GgufValueType::kFloat32) {
              float scalar = 0;
              if (!ReadPod(data_, size_, offset, scalar))
                return false;
              value = scalar;
            } else if (!ReadPod(data_, size_, offset, value)) {
              return false;
            }
            values.push_back(value);
          }
          meta_val.value = std::move(values);
        } else if (item_type == GgufValueType::kUint8 ||
                   item_type == GgufValueType::kUint16 ||
                   item_type == GgufValueType::kUint32 ||
                   item_type == GgufValueType::kBool ||
                   item_type == GgufValueType::kUint64) {
          std::vector<std::uint64_t> u_arr;
          u_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::uint64_t val = 0;
            if (item_type == GgufValueType::kUint8) {
              std::uint8_t val8 = 0;
              if (!ReadPod(data_, size_, offset, val8))
                return false;
              val = val8;
            } else if (item_type == GgufValueType::kUint16) {
              std::uint16_t val16 = 0;
              if (!ReadPod(data_, size_, offset, val16))
                return false;
              val = val16;
            } else if (item_type == GgufValueType::kUint32) {
              std::uint32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else if (item_type == GgufValueType::kBool) {
              std::uint8_t boolean = 0;
              if (!ReadPod(data_, size_, offset, boolean) || boolean > 1)
                return false;
              val = boolean;
            } else {
              if (!ReadPod(data_, size_, offset, val)) {
                return false;
              }
            }
            u_arr.push_back(val);
          }
          meta_val.value = std::move(u_arr);
        } else if (item_type == GgufValueType::kInt8 ||
                   item_type == GgufValueType::kInt16 ||
                   item_type == GgufValueType::kInt32 ||
                   item_type == GgufValueType::kInt64) {
          std::vector<std::int64_t> i_arr;
          i_arr.reserve(array_len);
          for (std::uint64_t a = 0; a < array_len; ++a) {
            std::int64_t val = 0;
            if (item_type == GgufValueType::kInt8) {
              std::int8_t val8 = 0;
              if (!ReadPod(data_, size_, offset, val8))
                return false;
              val = val8;
            } else if (item_type == GgufValueType::kInt16) {
              std::int16_t val16 = 0;
              if (!ReadPod(data_, size_, offset, val16))
                return false;
              val = val16;
            } else if (item_type == GgufValueType::kInt32) {
              std::int32_t val32 = 0;
              if (!ReadPod(data_, size_, offset, val32)) {
                return false;
              }
              val = val32;
            } else if (!ReadPod(data_, size_, offset, val)) {
              return false;
            }
            i_arr.push_back(val);
          }
          meta_val.value = std::move(i_arr);
        } else {
          return false;
        }
        break;
      }
    }

    if (key == "general.alignment") {
      const auto* value = std::get_if<std::uint64_t>(&meta_val.value);
      if (!value || !std::has_single_bit(*value) || *value > UINT32_MAX) {
        if (error_msg)
          *error_msg = "Invalid GGUF alignment";
        return false;
      }
      alignment_ = *value;
    }
    if (!metadata_.emplace(key, std::move(meta_val)).second) {
      if (error_msg)
        *error_msg = "Duplicate GGUF metadata key";
      return false;
    }
  }

  tensors_.reserve(tensor_count);
  tensor_index_.reserve(tensor_count);

  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    GgufTensorInfo info;
    if (!ReadString(data_, size_, offset, info.name)) {
      if (error_msg != nullptr) {
        *error_msg = "Failed to read tensor name";
      }
      return false;
    }

    if (info.name.empty() || info.name.size() > 65535 ||
        !charge(info.name.size(), 1))
      return false;
    std::uint32_t n_dims = 0;
    if (!ReadPod(data_, size_, offset, n_dims) || n_dims == 0 || n_dims > 8) {
      if (error_msg != nullptr) {
        *error_msg = "Invalid tensor dimensions count";
      }
      return false;
    }

    info.dimensions.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      if (!ReadPod(data_, size_, offset, info.dimensions[d])) {
        return false;
      }
    }

    std::uint32_t type_raw = 0;
    if (!ReadPod(data_, size_, offset, type_raw)) {
      return false;
    }
    if (type_raw > UINT16_MAX)
      return false;
    info.type = static_cast<GgmlType>(type_raw);
    const auto elements = info.ElementCount();
    const auto block = quant::QuantizedBlockElements(info.type);
    info.size_bytes = quant::EncodedSizeBytes(info.type, elements);
    if (!info.size_bytes || (block && info.dimensions.front() % block != 0)) {
      if (error_msg)
        *error_msg = "Invalid GGUF tensor shape or unsupported storage type: " +
                     std::string(info.name);
      return false;
    }

    if (!ReadPod(data_, size_, offset, info.offset)) {
      return false;
    }

    if (!tensor_index_.emplace(info.name, tensors_.size()).second) {
      if (error_msg)
        *error_msg = "Duplicate GGUF tensor name";
      return false;
    }
    tensors_.push_back(std::move(info));
  }

  const auto padding = (alignment_ - offset % alignment_) % alignment_;
  if (offset > size_ || padding > size_ - offset)
    return false;
  const std::size_t data_base = offset + padding;
  for (auto& tensor : tensors_) {
    if (tensor.offset % alignment_ != 0 || tensor.offset > size_ - data_base ||
        tensor.size_bytes > size_ - data_base - tensor.offset) {
      if (error_msg != nullptr) {
        *error_msg = "Tensor extent or alignment is invalid: " +
                     std::string(tensor.name);
      }
      return false;
    }
    tensor.data = data_ + data_base + tensor.offset;
  }

  return true;
}

const GgufMetadataValue* GgufReader::FindMetadata(
    std::string_view key) const noexcept {
  auto it = metadata_.find(key);
  if (it != metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::optional<std::string_view> GgufReader::GetMetadataString(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::string_view>(meta->value)) {
    return std::get<std::string_view>(meta->value);
  }
  return std::nullopt;
}

std::optional<std::uint32_t> GgufReader::GetMetadataUint32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  } else if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint32_t>(*value)) {
      return static_cast<std::uint32_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GgufReader::GetMetadataUint64(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<std::uint64_t>(&meta->value)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&meta->value)) {
    if (std::in_range<std::uint64_t>(*value)) {
      return static_cast<std::uint64_t>(*value);
    }
  }
  return std::nullopt;
}

std::optional<float> GgufReader::GetMetadataFloat32(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<double>(meta->value)) {
    return static_cast<float>(std::get<double>(meta->value));
  }
  return std::nullopt;
}

std::optional<bool> GgufReader::GetMetadataBool(
    std::string_view key) const noexcept {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr && std::holds_alternative<bool>(meta->value)) {
    return std::get<bool>(meta->value);
  }
  return std::nullopt;
}

std::vector<std::string_view> GgufReader::GetMetadataStringArray(
    std::string_view key) const {
  const auto* meta = FindMetadata(key);
  if (meta != nullptr &&
      std::holds_alternative<std::vector<std::string_view>>(meta->value)) {
    return std::get<std::vector<std::string_view>>(meta->value);
  }
  return {};
}

const GgufTensorInfo* GgufReader::FindTensor(
    std::string_view name) const noexcept {
  auto it = tensor_index_.find(name);
  if (it != tensor_index_.end()) {
    return &tensors_[it->second];
  }
  return nullptr;
}

bool GgufReader::HasTensor(std::string_view name) const noexcept {
  return tensor_index_.contains(name);
}

}  // namespace gufo::core
