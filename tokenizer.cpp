#include "tokenizer.hpp"

#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <queue>
#include <stdexcept>

#include "chat_template.hpp"

namespace gufo::models::lfm3 {
namespace {

constexpr std::string_view kEndOfText = "<|endoftext|>";
constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";

std::string FormatHexByteToken(std::uint8_t byte_val) {
  char buf[8];
  static_cast<void>(std::snprintf(buf, sizeof(buf), "<0x%02X>", byte_val));
  return {buf};
}

constexpr int HexCharToInt(char c) noexcept {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}

std::string ByteToGpt2Utf8(std::uint8_t b) {
  static const auto b2u_map = []() {
    std::array<std::string, 256> table;
    for (int i = '!'; i <= '~'; ++i)
      table[i] = std::string(1, static_cast<char>(i));
    for (int i = 161; i <= 172; ++i) {
      char buf[3] = {static_cast<char>(0xC0 | (i >> 6)),
                     static_cast<char>(0x80 | (i & 0x3F)), 0};
      table[i] = buf;
    }
    for (int i = 174; i <= 255; ++i) {
      char buf[3] = {static_cast<char>(0xC0 | (i >> 6)),
                     static_cast<char>(0x80 | (i & 0x3F)), 0};
      table[i] = buf;
    }
    int n = 0;
    for (int i = 0; i < 256; ++i) {
      if ((i < '!' || i > '~') && (i < 161 || i > 172) &&
          (i < 174 || i > 255)) {
        const int cp = 256 + n;
        char buf[3] = {static_cast<char>(0xC0 | (cp >> 6)),
                       static_cast<char>(0x80 | (cp & 0x3F)), 0};
        table[i] = buf;
        ++n;
      }
    }
    return table;
  }();
  return b2u_map[b];
}

std::string UnescapeGpt2Bytes(std::string_view text) {
  static const auto u2b_map = []() {
    std::unordered_map<char32_t, std::uint8_t> m;
    for (int b = '!'; b <= '~'; ++b)
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    for (int b = 161; b <= 172; ++b)
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    for (int b = 174; b <= 255; ++b)
      m[static_cast<char32_t>(b)] = static_cast<std::uint8_t>(b);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if ((b < '!' || b > '~') && (b < 161 || b > 172) &&
          (b < 174 || b > 255)) {
        m[static_cast<char32_t>(256 + n)] = static_cast<std::uint8_t>(b);
        ++n;
      }
    }
    return m;
  }();

  std::string result;
  result.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    const auto b0 = static_cast<unsigned char>(text[i]);
    char32_t cp = b0;
    std::size_t len = 1;
    if ((b0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp =
          ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
      len = 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = ((b0 & 0x0F) << 12) |
           ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 2]) & 0x3F);
      len = 3;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
      cp = ((b0 & 0x07) << 18) |
           ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 3]) & 0x3F);
      len = 4;
    }
    auto it = u2b_map.find(cp);
    if (it != u2b_map.end())
      result.push_back(static_cast<char>(it->second));
    else
      result.append(text.substr(i, len));
    i += len;
  }
  return result;
}

bool IsRecognizedSpecialSpelling(std::string_view t) {
  return t == kEndOfText || t == kImStart || t == kImEnd ||
         t == "<tool_call>" || t == "</tool_call>" || t == "<tool_response>" ||
         t == "</tool_response>" || t == "<think>" || t == "</think>" ||
         t == "<tts_pad>" || t == "<tts_text_bos>" || t == "<tts_text_eod>" ||
         t == "<tts_text_bos_single>" || t == "<|object_ref_start|>" ||
         t == "<|object_ref_end|>" || t == "<|vision_start|>" ||
         t == "<|vision_end|>" || t == "<|image_pad|>" ||
         t == "<|video_pad|>" || t == "<|quad_start|>" || t == "<|quad_end|>" ||
         (t.size() >= 4 && t.starts_with("<|") && t.ends_with("|>"));
}

// LiquidAI tokenizer.json uses the Llama 3 / LFM2 split (digits in groups of
// three), followed by byte-level BPE. Qwen35 has a different split.
const icu::RegexPattern& Pattern() {
  static const auto pattern = [] {
    UErrorCode status = U_ZERO_ERROR;
    auto* p = icu::RegexPattern::compile(
        icu::UnicodeString::fromUTF8(
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"),
        0, status);
    if (U_FAILURE(status))
      throw std::runtime_error("invalid LFM3 tokenizer regex");
    return std::unique_ptr<icu::RegexPattern>(p);
  }();
  return *pattern;
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(const core::GgufReader& reader,
                                           std::string* error) {
  if (reader.GetMetadataString("tokenizer.ggml.pre") != "lfm2") {
    if (error)
      *error = "LFM3 requires tokenizer.ggml.pre=lfm2";
    return nullptr;
  }
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  if (tokens.empty()) {
    if (error)
      *error = "GGUF metadata missing 'tokenizer.ggml.tokens'";
    return nullptr;
  }
  const auto* entry = reader.FindMetadata("tokenizer.ggml.token_type");
  const auto* signed_types =
      entry ? std::get_if<std::vector<std::int64_t>>(&entry->value) : nullptr;
  const auto* unsigned_types =
      entry ? std::get_if<std::vector<std::uint64_t>>(&entry->value) : nullptr;
  if ((!signed_types || signed_types->size() != tokens.size()) &&
      (!unsigned_types || unsigned_types->size() != tokens.size())) {
    if (error)
      *error = "LFM3 requires per-token GGUF types";
    return nullptr;
  }

  auto result = std::make_unique<Tokenizer>();
  result->id_to_token_.reserve(tokens.size());
  result->token_to_id_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    result->id_to_token_.emplace_back(tokens[i]);
    result->token_to_id_[result->id_to_token_.back()] = static_cast<TokenId>(i);
  }

  const auto merges = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  result->merge_ranks_.reserve(merges.size());
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const auto merge = merges[rank];
    const auto space_pos = merge.find(' ');
    if (space_pos == std::string_view::npos)
      continue;
    const std::string part1(merge.substr(0, space_pos));
    const std::string part2(merge.substr(space_pos + 1));
    auto it1 = result->token_to_id_.find(part1);
    auto it2 = result->token_to_id_.find(part2);
    if (it1 != result->token_to_id_.end() &&
        it2 != result->token_to_id_.end()) {
      const auto merged = result->token_to_id_.find(part1 + part2);
      result->merge_ranks_[{it1->second, it2->second}] = {
          .rank = rank,
          .token = merged != result->token_to_id_.end() ? merged->second
                                                        : kInvalidTokenId,
      };
    }
  }

  result->special_pieces_.resize(tokens.size());
  result->control_.resize(tokens.size());
  result->id_to_decoded_token_.resize(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    auto type =
        signed_types ? std::uint64_t((*signed_types)[i]) : (*unsigned_types)[i];
    result->control_[i] = type == 3;
    const bool is_special = IsRecognizedSpecialSpelling(tokens[i]);
    if (type == 3 || type == 4 || is_special) {
      result->special_tokens_.emplace_back(std::string(tokens[i]),
                                           static_cast<std::uint32_t>(i));
      result->special_pieces_[i] = std::string(tokens[i]);
    }
    if (is_special) {
      result->id_to_decoded_token_[i] = result->id_to_token_[i];
    } else {
      const auto& tok = result->id_to_token_[i];
      if (tok.size() == 6 && tok.starts_with("<0x") && tok.ends_with('>')) {
        const int h1 = HexCharToInt(tok[3]);
        const int h2 = HexCharToInt(tok[4]);
        if (h1 >= 0 && h2 >= 0) {
          const auto byte_val = static_cast<std::uint8_t>((h1 << 4) | h2);
          result->id_to_decoded_token_[i] =
              std::string(1, static_cast<char>(byte_val));
          continue;
        }
      }
      result->id_to_decoded_token_[i] = UnescapeGpt2Bytes(tok);
    }
  }

  for (std::size_t b = 0; b < 256; ++b) {
    const auto byte_val = static_cast<std::uint8_t>(b);
    const std::string gpt2_utf8 = ByteToGpt2Utf8(byte_val);
    if (auto it = result->token_to_id_.find(gpt2_utf8);
        it != result->token_to_id_.end()) {
      result->byte_tokens_[b] = it->second;
    } else if (auto direct_it = result->token_to_id_.find(
                   std::string(1, static_cast<char>(byte_val)));
               direct_it != result->token_to_id_.end()) {
      result->byte_tokens_[b] = direct_it->second;
    } else if (auto hex_it =
                   result->token_to_id_.find(FormatHexByteToken(byte_val));
               hex_it != result->token_to_id_.end()) {
      result->byte_tokens_[b] = hex_it->second;
    } else {
      result->byte_tokens_[b] = kInvalidTokenId;
    }
  }

  if (auto it = result->token_to_id_.find(std::string(kEndOfText));
      it != result->token_to_id_.end())
    result->endoftext_token_id_ = it->second;
  if (auto eos = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    result->eos_token_id_ = *eos;
  } else if (auto it = result->token_to_id_.find(std::string(kImEnd));
             it != result->token_to_id_.end()) {
    result->eos_token_id_ = it->second;
  } else {
    result->eos_token_id_ = result->endoftext_token_id_;
  }
  if (auto bos = reader.GetMetadataUint32("tokenizer.ggml.bos_token_id")) {
    result->bos_token_id_ = *bos;
  } else if (auto it = result->token_to_id_.find(std::string(kImStart));
             it != result->token_to_id_.end()) {
    result->bos_token_id_ = it->second;
  }
  return result;
}

std::vector<std::uint32_t> Tokenizer::BpeMergeChunk(
    std::string_view chunk) const {
  if (chunk.empty())
    return {};
  std::vector<TokenId> word_tokens;
  word_tokens.reserve(chunk.size());
  for (const unsigned char c : chunk) {
    const TokenId tid = byte_tokens_[c];
    if (tid != kInvalidTokenId)
      word_tokens.push_back(tid);
  }
  if (word_tokens.size() <= 1)
    return word_tokens;

  constexpr std::size_t kNoSymbol = std::numeric_limits<std::size_t>::max();
  const std::size_t count = word_tokens.size();
  std::vector<std::size_t> previous(count);
  std::vector<std::size_t> following(count);
  for (std::size_t i = 0; i < count; ++i) {
    previous[i] = i == 0 ? kNoSymbol : i - 1;
    following[i] = i + 1 < count ? i + 1 : kNoSymbol;
  }

  struct Candidate {
    std::uint32_t rank;
    std::size_t index;
    TokenId left;
    TokenId right;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)>
      candidates(later);

  const auto offer = [&](std::size_t index, std::size_t right) {
    const auto merge =
        merge_ranks_.find({word_tokens[index], word_tokens[right]});
    if (merge != merge_ranks_.end()) {
      candidates.push({
          .rank = merge->second.rank,
          .index = index,
          .left = word_tokens[index],
          .right = word_tokens[right],
      });
    }
  };
  for (std::size_t i = 0; i + 1 < count; ++i)
    offer(i, i + 1);

  while (!candidates.empty()) {
    const Candidate best = candidates.top();
    candidates.pop();
    const std::size_t right = following[best.index];
    if (word_tokens[best.index] != best.left || right == kNoSymbol ||
        word_tokens[right] != best.right)
      continue;
    const auto merge = merge_ranks_.find({best.left, best.right});
    if (merge == merge_ranks_.end() || merge->second.token == kInvalidTokenId)
      break;
    word_tokens[best.index] = merge->second.token;
    const std::size_t after = following[right];
    following[best.index] = after;
    if (after != kNoSymbol)
      previous[after] = best.index;
    word_tokens[right] = kInvalidTokenId;
    if (previous[best.index] != kNoSymbol)
      offer(previous[best.index], best.index);
    if (after != kNoSymbol)
      offer(best.index, after);
  }

  std::vector<TokenId> merged;
  merged.reserve(count);
  for (std::size_t i = 0; i != kNoSymbol; i = following[i])
    merged.push_back(word_tokens[i]);
  return merged;
}

std::vector<std::uint32_t> Tokenizer::Encode(std::string_view text,
                                             bool special) const {
  return EncodeImpl(text, special, {});
}

std::vector<std::uint32_t> Tokenizer::EncodeFramed(
    std::string_view text,
    std::span<const std::pair<std::size_t, std::size_t>> literals) const {
  return EncodeImpl(text, true, literals);
}

std::vector<std::uint32_t> Tokenizer::EncodeImpl(
    std::string_view text, bool special,
    std::span<const std::pair<std::size_t, std::size_t>> literals) const {
  std::vector<std::uint32_t> ids;
  auto regular = [&](std::string_view part) {
    auto unicode = icu::UnicodeString::fromUTF8(
        icu::StringPiece(part.data(), part.size()));
    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> matcher(
        Pattern().matcher(unicode, status));
    if (U_FAILURE(status))
      throw std::runtime_error("LFM3 tokenizer regex failed");
    while (matcher->find(status)) {
      std::string chunk;
      matcher->group(status).toUTF8String(chunk);
      auto row = BpeMergeChunk(chunk);
      ids.insert(ids.end(), row.begin(), row.end());
    }
    if (U_FAILURE(status))
      throw std::runtime_error("LFM3 tokenizer matching failed");
  };
  if (!special) {
    regular(text);
    return ids;
  }
  const auto literal_overlap = [&](std::size_t start, std::size_t size) {
    const auto it = std::lower_bound(literals.begin(), literals.end(), start,
                                     [](const auto& span, std::size_t pos) {
                                       return span.first + span.second <= pos;
                                     });
    return it != literals.end() && it->first < start + size;
  };
  const auto next = [&](std::string_view spelling, std::size_t from) {
    auto at = text.find(spelling, from);
    while (at != text.npos && literal_overlap(at, spelling.size()))
      at = text.find(spelling, at + 1);
    return at;
  };
  // Each spelling scans the input once, including long histories with many
  // native tags. Later control occurrences stay cached until consumed.
  std::vector<std::size_t> occurrence(special_tokens_.size());
  for (std::size_t i = 0; i < occurrence.size(); ++i)
    occurrence[i] = next(special_tokens_[i].first, 0);
  std::size_t offset = 0;
  while (offset < text.size()) {
    std::size_t first = text.size(), length = 0;
    std::uint32_t token = 0;
    for (std::size_t i = 0; i < occurrence.size(); ++i) {
      const auto& [spelling, id] = special_tokens_[i];
      if (occurrence[i] < offset)
        occurrence[i] = next(spelling, offset);
      auto at = occurrence[i];
      if (at < first || (at == first && spelling.size() > length)) {
        first = at;
        length = spelling.size();
        token = id;
      }
    }
    regular(text.substr(offset, first - offset));
    if (!length)
      break;
    ids.push_back(token);
    offset = first + length;
  }
  return ids;
}

std::string Tokenizer::Render(
    std::span<const std::pair<std::string, std::string>> messages,
    bool thinking) {
  std::vector<ext::ChatMessage> chat;
  for (const auto& [role, content] : messages) {
    const auto kind =
        role == "system" ? ext::ChatRole::kSystem
        : role == "user" ? ext::ChatRole::kUser
        : role == "assistant"
            ? ext::ChatRole::kAssistant
            : throw std::invalid_argument("unsupported LFM3 message role");
    chat.emplace_back(kind, content);
  }
  ReasoningOptions options;
  options.enabled = thinking;
  return RenderChat(chat, {}, options);
}

}  // namespace gufo::models::lfm3
