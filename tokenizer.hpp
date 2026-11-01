#ifndef GUFO_MODELS_LFM3_TOKENIZER_HPP_
#define GUFO_MODELS_LFM3_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gufo::models::lfm3 {

class Tokenizer {
public:
  using TokenId = std::uint32_t;
  static constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

  static std::unique_ptr<Tokenizer> Load(const core::GgufReader& reader,
                                         std::string* error = nullptr);
  std::vector<std::uint32_t> Encode(std::string_view text,
                                    bool special = true) const;
  // Literal spans suppress control-token matching without introducing BPE
  // boundaries between client text and adjacent template whitespace.
  std::vector<std::uint32_t> EncodeFramed(
      std::string_view text,
      std::span<const std::pair<std::size_t, std::size_t>> literals) const;
  std::string Decode(std::span<const std::uint32_t> tokens) const {
    std::string text;
    for (auto token : tokens)
      text += Piece(token);
    return text;
  }
  std::string_view Piece(std::uint32_t token) const {
    if (token < special_pieces_.size() && !special_pieces_[token].empty())
      return special_pieces_[token];
    if (token < id_to_decoded_token_.size())
      return id_to_decoded_token_[token];
    if (token < id_to_token_.size())
      return id_to_token_[token];
    return "";
  }
  bool IsStop(std::uint32_t token) const {
    return (eos_token_id_ != kInvalidTokenId && token == eos_token_id_) ||
           (endoftext_token_id_ != kInvalidTokenId &&
            token == endoftext_token_id_);
  }
  // GGUF CONTROL tokens (special in the Hugging Face tokenizer), which
  // decoding with skip_special_tokens drops.
  bool IsControl(std::uint32_t token) const {
    return token < control_.size() && control_[token];
  }
  std::uint32_t Bos() const { return bos_token_id_; }
  static std::string Render(
      std::span<const std::pair<std::string, std::string>> messages,
      bool thinking = true);

private:
  struct PairHash {
    std::size_t operator()(
        const std::pair<TokenId, TokenId>& p) const noexcept {
      return (static_cast<std::size_t>(p.first) << 32) |
             static_cast<std::size_t>(p.second);
    }
  };
  struct MergeRule {
    std::uint32_t rank;
    TokenId token;
  };

  std::vector<std::uint32_t> BpeMergeChunk(std::string_view chunk) const;
  std::vector<std::uint32_t> EncodeImpl(
      std::string_view text, bool special,
      std::span<const std::pair<std::size_t, std::size_t>> literals) const;

  std::vector<std::string> id_to_token_;
  std::vector<std::string> id_to_decoded_token_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::pair<TokenId, TokenId>, MergeRule, PairHash>
      merge_ranks_;
  std::array<TokenId, 256> byte_tokens_{};
  TokenId bos_token_id_{kInvalidTokenId};
  TokenId eos_token_id_{kInvalidTokenId};
  TokenId endoftext_token_id_{kInvalidTokenId};
  std::vector<std::pair<std::string, std::uint32_t>> special_tokens_;
  std::vector<std::string> special_pieces_;
  std::vector<bool> control_;
};

}  // namespace gufo::models::lfm3
#endif
