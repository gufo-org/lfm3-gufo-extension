#include <algorithm>
#include <chrono>
#include <gufo/ext/text.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.hpp"

namespace gufo::models::lfm3 {
namespace {
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

int RunLfm3PromptImpl(const ext::TextPromptOptions& opt, bool interactive,
                      Clock::time_point load_start) {
  try {
    if (opt.force_cpu || !opt.image_paths.empty() ||
        !opt.vision_model_path.empty() || !opt.speculative_backend.empty())
      throw std::runtime_error(
          "LFM3 supports autoregressive HIP text inference");
    if (opt.reasoning_mode != "auto" && opt.reasoning_mode != "on" &&
        opt.reasoning_mode != "off")
      throw std::runtime_error("LFM3 reasoning must be auto, on, or off");
    if (opt.reasoning_effort != "auto" || opt.preserve_thinking != "auto")
      throw std::runtime_error(
          "LFM3 does not support reasoning effort or preserve-thinking "
          "overrides");
    std::string error;
    const auto load =
        load_start == Clock::time_point{} ? Clock::now() : load_start;
    auto model = Model::Load(opt.model_path, &error);
    Require(bool(model), error);
    std::cerr << "[ModelLoad]: " << Seconds(load) << " s\n";
    auto session = model->CreateSession(
        std::min<std::uint32_t>(8192, model->config().context), &error);
    Require(bool(session), error);
    const auto& tokenizer = model->tokenizer();
    std::vector<std::pair<std::string, std::string>> history;
    if (!opt.system_prompt.empty())
      history.emplace_back("system", opt.system_prompt);
    std::vector<std::uint32_t> processed;
    std::vector<float> logits(model->config().vocab);
    std::string input = opt.prompt_text;
    do {
      if (interactive) {
        std::cout << ">>> User: " << std::flush;
        if (!std::getline(std::cin, input) || input == "exit")
          break;
      }
      history.emplace_back("user", input);
      const auto rendered =
          opt.use_chat_template
              ? Tokenizer::Render(history, opt.reasoning_mode != "off")
              : input;
      auto tokens = tokenizer.Encode(rendered);
      if (tokens.empty())
        tokens.push_back(tokenizer.Bos());
      const bool prefix =
          tokens.size() > processed.size() &&
          std::equal(processed.begin(), processed.end(), tokens.begin());
      if (!prefix) {
        Require(session->Reset(&error), error);
        processed.clear();
      }
      const auto prefill = Clock::now();
      Require(session->Forward(std::span(tokens).subspan(processed.size()),
                               logits, &error),
              error);
      const auto pp = Seconds(prefill);
      processed = tokens;
      sampling::SamplerState sampler(opt.sampling, tokens);
      std::vector<std::uint32_t> generated;
      const auto decode = Clock::now();
      for (std::size_t i = 0; i < opt.max_tokens; ++i) {
        const auto token = sampler.Sample(logits);
        if (tokenizer.IsStop(token))
          break;
        generated.push_back(token);
        sampler.Accept(token);
        std::cout << tokenizer.Piece(token) << std::flush;
        if (session->Position() >= 8192)
          break;
        Require(session->Forward(std::span(&token, 1), logits, &error), error);
        processed.push_back(token);
      }
      const auto tg = Seconds(decode);
      std::cout << '\n';
      if (opt.verbose)
        std::cerr << "LFM3 pp " << tokens.size() << " tokens " << pp
                  << " s; tg " << generated.size() << " tokens " << tg << " s ("
                  << generated.size() / tg << " tok/s)\n";
      history.emplace_back("assistant", tokenizer.Decode(generated));
    } while (interactive);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "LFM3: " << e.what() << '\n';
    return 1;
  }
}
}  // namespace

int RunLfm3Prompt(const ext::TextPromptOptions& opt, const core::GgufReader&,
                  std::chrono::steady_clock::time_point load_start) {
  return RunLfm3PromptImpl(opt, false, load_start);
}

int RunLfm3Chat(const ext::TextPromptOptions& opt, const core::GgufReader&,
                std::chrono::steady_clock::time_point load_start) {
  return RunLfm3PromptImpl(opt, true, load_start);
}

int RunLfm3Bench(const ext::TextBenchOptions& opt,
                 const std::shared_ptr<const core::GgufReader>&,
                 std::chrono::steady_clock::time_point) {
  try {
    if (opt.concurrency != std::vector<std::size_t>{1} ||
        !opt.speculative_backend.empty() || opt.enable_yarn ||
        opt.validate_prefill_tokens)
      throw std::runtime_error(
          "LFM3 bench supports C1 autoregressive timing; use lfm3_probe for "
          "logit validation");
    std::string error;
    auto model = Model::Load(opt.model_path, &error);
    Require(bool(model), error);
    auto max = [](const auto& v) {
      return v.empty() ? std::size_t(0) : *std::max_element(v.begin(), v.end());
    };
    auto context =
        max(opt.n_depths) + std::max(max(opt.n_prompts), max(opt.n_gens)) + 1;
    if (context > model->config().context)
      throw std::runtime_error("LFM3 benchmark exceeds model context");
    auto session = model->CreateSession(context, &error);
    Require(bool(session), error);
    std::vector<float> logits(model->config().vocab);
    auto seed = model->tokenizer().Encode(
        "The quick brown fox jumps over the lazy dog. ");
    auto fill = [&](std::size_t n) {
      std::vector<std::uint32_t> ids(n);
      for (std::size_t i = 0; i < n; ++i)
        ids[i] = seed[i % seed.size()];
      return ids;
    };
    Require(session->Forward(seed, logits, &error),
            error);  // Warm both prompt and decode routes.
    auto token = std::uint32_t(std::max_element(logits.begin(), logits.end()) -
                               logits.begin());
    Require(session->Forward(std::span(&token, 1), logits, &error), error);
    std::cout << "| model | test | depth | repetition | tokens/s | seconds "
                 "|\n| :-- | :-- | --: | --: | --: | --: |\n";
    for (auto depth : opt.n_depths) {
      auto prepare = [&] {
        Require(session->Reset(&error), error);
        if (depth)
          Require(session->Forward(fill(depth), {}, &error), error);
      };
      // Warm the actual prompt shapes, including a full 128-token chunk,
      // before any timed request. Session scratch is already allocated.
      for (auto n : opt.n_prompts) {
        if (!n)
          continue;
        prepare();
        Require(session->Forward(fill(n), {}, &error), error);
      }
      for (std::size_t r = 0; r < opt.repetitions; ++r) {
        for (auto n : opt.n_prompts) {
          if (!n)
            continue;
          prepare();
          auto prompt = fill(n);
          const auto start = Clock::now();
          Require(session->Forward(prompt, {}, &error), error);
          const auto dt = Seconds(start);
          std::cout << "| LFM3-1.7B | pp" << n << " | " << depth << " | " << r
                    << " | " << n / dt << " | " << dt << " |\n";
        }
        for (auto n : opt.n_gens) {
          if (!n)
            continue;
          prepare();
          auto prompt = fill(1);
          Require(session->Forward(prompt, logits, &error), error);
          sampling::SamplerState sampler(opt.sampling, prompt);
          const auto start = Clock::now();
          for (std::size_t i = 0; i < n; ++i) {
            token = sampler.Sample(logits);
            sampler.Accept(token);
            Require(session->Forward(std::span(&token, 1), logits, &error),
                    error);
          }
          const auto dt = Seconds(start);
          std::cout << "| LFM3-1.7B | tg" << n << " | " << depth << " | " << r
                    << " | " << n / dt << " | " << dt << " |\n";
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "LFM3: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace gufo::models::lfm3
