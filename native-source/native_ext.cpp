// SPDX-License-Identifier: Apache-2.0
// Adapted from T8mars/comfyui-confucius-r2t2-t8 at b317076ac8f8aee3e5ecb253b94fa389fed2d346.
// T8 Drama modification: C ABI instead of pybind11, cancellation and context limits.
// Inference remains in the caller process; no Python ABI-specific extension.
// See vendor/r2t2_native/NOTICE.md and LICENSE for provenance and terms.


#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"  // decode_embd_batch: project-internal mtmd helper used for external embeddings

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>



namespace {

void private_worker_log(enum ggml_log_level level, const char * message, void *) {
    // mtmd INFO logs can contain the entire prompt and partial transcript.
    if (level == GGML_LOG_LEVEL_ERROR && message) {
        std::fputs(message, stderr);
    }
}

struct ModelDeleter {
    void operator()(llama_model * model) const {
        if (model) {
            llama_model_free(model);
        }
    }
};

struct ContextDeleter {
    void operator()(llama_context * ctx) const {
        if (ctx) {
            llama_free(ctx);
        }
    }
};

struct MtmdDeleter {
    void operator()(mtmd_context * ctx) const {
        if (ctx) {
            mtmd_free(ctx);
        }
    }
};

struct BitmapDeleter {
    void operator()(mtmd_bitmap * bitmap) const {
        if (bitmap) {
            mtmd_bitmap_free(bitmap);
        }
    }
};

struct ChunksDeleter {
    void operator()(mtmd_input_chunks * chunks) const {
        if (chunks) {
            mtmd_input_chunks_free(chunks);
        }
    }
};

struct SamplerDeleter {
    void operator()(llama_sampler * sampler) const {
        if (sampler) {
            llama_sampler_free(sampler);
        }
    }
};

using model_ptr   = std::unique_ptr<llama_model, ModelDeleter>;
using context_ptr = std::unique_ptr<llama_context, ContextDeleter>;
using mtmd_ptr    = std::unique_ptr<mtmd_context, MtmdDeleter>;
using bitmap_ptr  = std::unique_ptr<mtmd_bitmap, BitmapDeleter>;
using chunks_ptr  = std::unique_ptr<mtmd_input_chunks, ChunksDeleter>;
using sampler_ptr = std::unique_ptr<llama_sampler, SamplerDeleter>;

std::string replace_all(std::string value, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
    return value;
}

// Optional conservative streaming rule copied from the hybrid implementation.
// With QWEN3ASR_STOP_BIAS unset (or <= 0), this is exactly greedy argmax.
float stop_bias_threshold() {
    static const float threshold = []() {
        const char * value = std::getenv("QWEN3ASR_STOP_BIAS");
        if (!value || !*value) {
            return 0.0f;
        }
        return static_cast<float>(std::atof(value));
    }();
    return threshold;
}

bool is_stop_like(llama_token token) {
    // 91 = '|', 2 = '#', 151645 = <|im_end|>
    return token == 91 || token == 2 || token == 151645;
}

llama_token pick_token_greedy(const float * logits, int32_t n_vocab) {
    // QWEN3ASR_STOP_BIAS 没有设置时，阈值为0，pick_token_greedy会完全退化为普通argmax，不会执行偏向停止token的逻辑
    // pick_token_greedy可以处理如下情况：
    // top1 = 普通文字 token
    // top2 = | 或 # 或 <|im_end|> (stop)
    // top1 与 top2 的 logit 差小于阈值
    // 且QWEN3ASR_STOP_BIAS=0.1
    // 则会选择top2
    llama_token best = 0;
    llama_token second = -1;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            second = best;
            best = i;
        } else if (second < 0 || logits[i] > logits[second]) {
            second = i;
        }
    }
    const float threshold = stop_bias_threshold();
    if (threshold > 0.0f && second >= 0 && !is_stop_like(best) && is_stop_like(second) &&
            logits[best] - logits[second] < threshold) {
        return second;
    }
    return best;
}

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    char buffer[512];
    for (const llama_token token : tokens) {
        const int32_t n = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
        if (n > 0) {
            result.append(buffer, static_cast<size_t>(n));
        }
    }
    return result;
}

uint64_t fnv1a_f32(const float * values, size_t count) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(values);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool debug_enabled() {
    const char * value = std::getenv("DEBUG_PRINT");
    return value != nullptr && (value[0] == '1' || value[0] == 'y' || value[0] == 'Y');
}

bool embed_bf16_enabled() {
    const char * value = std::getenv("QWEN3ASR_EMBED_BF16");
    return value != nullptr && (value[0] == '1' || value[0] == 'y' || value[0] == 'Y');
}

float round_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits &= 0xffff0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void dump_f32_matrix(const char * path, const float * values, int64_t n_rows, int64_t n_cols) {
    if (!path || !values || n_rows <= 0 || n_cols <= 0) {
        return;
    }
    FILE * file = std::fopen(path, "wb");
    if (!file) {
        std::fprintf(stderr, "[DEBUG-DUMP][llama] failed to open %s\n", path);
        return;
    }
    // Header: magic, rows, cols. Payload is row-major float32.
    const uint32_t header[3] = { 0x51454d42U, (uint32_t)n_rows, (uint32_t)n_cols };
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(values, sizeof(float), (size_t)n_rows * (size_t)n_cols, file);
    std::fclose(file);
    std::fprintf(stderr, "[DEBUG-DUMP][llama] embedding path=%s shape=[%lld,%lld]\n",
        path, (long long)n_rows, (long long)n_cols);
}

std::vector<std::pair<int32_t, float>> topk_logits(const float * logits, int32_t n_vocab, int k) {
    std::vector<int32_t> indices((size_t)n_vocab);
    std::iota(indices.begin(), indices.end(), 0);
    k = std::max(1, std::min(k, n_vocab));
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [logits](int32_t lhs, int32_t rhs) { return logits[lhs] > logits[rhs]; });
    std::vector<std::pair<int32_t, float>> result;
    result.reserve((size_t)k);
    for (int i = 0; i < k; ++i) {
        result.emplace_back(indices[(size_t)i], logits[indices[(size_t)i]]);
    }
    return result;
}

std::vector<std::pair<int32_t, float>> debug_first_logits(
        llama_context * ctx, const llama_vocab * vocab) {
    std::vector<std::pair<int32_t, float>> result;
    if (!debug_enabled() || !ctx || !vocab) {
        return result;
    }
    // -1 selects the final output position (the first next-token logits after
    // the complete audio+text prompt has been prefetched).
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (!logits || n_vocab <= 0) {
        std::fprintf(stderr, "[DEBUG-LOGITS][llama] logits unavailable\n");
        return result;
    }
    int k = 10;
    if (const char * value = std::getenv("QWEN3ASR_LOGITS_TOPK")) {
        k = std::atoi(value);
    }
    result = topk_logits(logits, n_vocab, k);
    std::fprintf(stderr, "[DEBUG-LOGITS][llama] first_position_topk=");
    for (size_t i = 0; i < result.size(); ++i) {
        std::fprintf(stderr, "%s{\"id\":%d,\"logit\":%.9g}",
            i == 0 ? "[" : ",", result[i].first, result[i].second);
    }
    std::fprintf(stderr, "]\n");
    return result;
}

void debug_embedding(const float * values, int64_t n_tokens, int64_t n_embd, size_t chunk_index) {
    if (!debug_enabled() || values == nullptr || n_tokens <= 0 || n_embd <= 0) {
        return;
    }
    const size_t count = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd);
    double sum = 0.0;
    double sum_sq = 0.0;
    float min_value = values[0];
    float max_value = values[0];
    for (size_t i = 0; i < count; ++i) {
        const float value = values[i];
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    std::fprintf(stderr,
        "[DEBUG-EMB][llama] chunk=%zu shape=[%lld,%lld] min=%.9g max=%.9g mean=%.9g rms=%.9g checksum=0x%016llx first8=",
        chunk_index, (long long)n_tokens, (long long)n_embd,
        min_value, max_value, sum / count, std::sqrt(sum_sq / count),
        (unsigned long long)fnv1a_f32(values, count));
    const int64_t n_first = std::min<int64_t>(8, n_embd);
    for (int64_t i = 0; i < n_first; ++i) {
        std::fprintf(stderr, "%.9g%s", values[i], i + 1 == n_first ? "" : ",");
    }
    std::fprintf(stderr, "\n");
}

struct DecodeResult { std::string text; bool truncated; };
using cancel_callback = bool (*)(void *);

class Qwen3ASRNative {
public:
    Qwen3ASRNative(
            const std::string & model_path,
            const std::string & mmproj_path,
            int n_ctx = 32768,
            int n_batch = 8192,
            int n_threads = 32,
            bool use_gpu = true,
            int n_gpu_layers = -1) {
        ggml_log_set(private_worker_log, nullptr);
        llama_log_set(private_worker_log, nullptr);
        mtmd_log_set(private_worker_log, nullptr);
        n_batch_ = n_batch;
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        model_.reset(llama_model_load_from_file(model_path.c_str(), model_params));
        if (!model_) {
            throw std::runtime_error("failed to load llama model: " + model_path);
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(n_ctx);
        context_params.n_batch = static_cast<uint32_t>(n_batch);
        context_params.n_ubatch = static_cast<uint32_t>(std::min(n_batch, 512));
        context_params.n_seq_max = 1;
        context_params.n_threads = n_threads;
        context_params.n_threads_batch = n_threads;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_params.offload_kqv = use_gpu;
        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) {
            throw std::runtime_error("failed to create llama context");
        }

        mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = use_gpu;
        mtmd_params.n_threads = n_threads;
        mtmd_params.warmup = false;
        mtmd_.reset(mtmd_init_from_file(mmproj_path.c_str(), model_.get(), mtmd_params));
        if (!mtmd_) {
            throw std::runtime_error("failed to load mtmd projector: " + mmproj_path);
        }
        if (!mtmd_support_audio(mtmd_.get())) {
            throw std::runtime_error("the mtmd projector does not support audio");
        }
    }

    // Tokenize text with the vocabulary embedded in the loaded GGUF model.
    // This is used by the streaming prefix-rollback path so that the token
    // boundaries match the tokenizer used by llama.cpp during generation.
    std::vector<llama_token> tokenize(
            const std::string & text,
            bool add_special = false,
            bool parse_special = true) const {
        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        if (!vocab) {
            throw std::runtime_error("llama model has no vocabulary");
        }
        if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("text is too large to tokenize");
        }

        // llama_tokenize() returns a negative value when the supplied buffer
        // is too small; the absolute value is the required token count.
        const size_t initial_capacity = std::max<size_t>(2, text.size() + 2);
        std::vector<llama_token> tokens(initial_capacity);
        int32_t count = llama_tokenize(
                vocab,
                text.data(),
                static_cast<int32_t>(text.size()),
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                add_special,
                parse_special);

        if (count == std::numeric_limits<int32_t>::min()) {
            throw std::runtime_error("llama_tokenize failed: result exceeds int32_t limit");
        }
        if (count < 0) {
            const size_t required = static_cast<size_t>(-count);
            tokens.resize(required);
            count = llama_tokenize(
                    vocab,
                    text.data(),
                    static_cast<int32_t>(text.size()),
                    tokens.data(),
                    static_cast<int32_t>(tokens.size()),
                    add_special,
                    parse_special);
        }
        if (count < 0) {
            throw std::runtime_error("llama_tokenize failed with code " + std::to_string(count));
        }

        tokens.resize(static_cast<size_t>(count));
        return tokens;
    }

    // Detokenize token IDs with the vocabulary embedded in the loaded GGUF.
    // `unparse_special=true` preserves special/control-token text, which is
    // needed when a streaming prefix contains markers such as <asr_text>.
    std::string detokenize_tokens(
            const std::vector<llama_token> & tokens,
            bool unparse_special = true) const {
        if (tokens.empty()) {
            return {};
        }

        const llama_vocab * vocab = llama_model_get_vocab(model_.get());
        if (!vocab) {
            throw std::runtime_error("llama model has no vocabulary");
        }
        if (tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("too many tokens to detokenize");
        }

        // The API reports the required size as a negative value when this
        // buffer is too small, so retry with the exact size in that case.
        size_t initial_capacity = std::max<size_t>(64, tokens.size() * 8);
        initial_capacity = std::min(
                initial_capacity,
                static_cast<size_t>(std::numeric_limits<int32_t>::max()));
        std::string text(initial_capacity, '\0');
        int32_t count = llama_detokenize(
                vocab,
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                text.data(),
                static_cast<int32_t>(text.size()),
                false,             // remove_special
                unparse_special);

        if (count < 0) {
            if (count == std::numeric_limits<int32_t>::min()) {
                throw std::runtime_error("llama_detokenize failed: result exceeds int32_t limit");
            }
            text.resize(static_cast<size_t>(-count));
            count = llama_detokenize(
                    vocab,
                    tokens.data(),
                    static_cast<int32_t>(tokens.size()),
                    text.data(),
                    static_cast<int32_t>(text.size()),
                    false,
                    unparse_special);
        }
        if (count < 0) {
            throw std::runtime_error("llama_detokenize failed with code " + std::to_string(count));
        }

        text.resize(static_cast<size_t>(count));
        return text;
    }

    DecodeResult generate_once(
            const float * audio_data, size_t audio_size, cancel_callback cancel,
            const std::string & prompt,
            int max_tokens = 4096) {
        // 音频 encoder，llm 推理
        if (!audio_data || audio_size == 0) {
            throw std::invalid_argument("audio must be a non-empty 1-D float32 array");
        }
        if (max_tokens <= 0) {
            throw std::invalid_argument("max_tokens must be positive");
        }



        std::string generated_text;
        std::vector<llama_token> generated;
        std::vector<llama_token> prompt_text_tokens;
        std::vector<size_t> media_token_counts;
        std::vector<llama_pos> media_position_counts;
        std::vector<float> all_embeddings;
        std::vector<float> all_decode_embeddings;
        const int64_t embedding_dim = llama_model_n_embd_inp(model_.get());
        const char * embedding_dump_path = std::getenv("QWEN3ASR_EMBED_DUMP_PATH");
        const char * decode_embedding_dump_path = std::getenv("QWEN3ASR_EMBED_DECODE_DUMP_PATH");
        size_t prompt_token_count = 0;
        llama_pos prompt_position_count = 0;
        size_t chunk_count = 0;
        std::string finish_reason;
        std::vector<std::pair<int32_t, float>> first_logits_topk;
        {
            if (cancel && cancel(nullptr)) throw std::runtime_error("ASR_CANCELLED");

            llama_memory_clear(llama_get_memory(context_.get()), true);

            // vLLM uses the Qwen3-ASR token triplet. mtmd consumes its own media marker;
            // the marker is expanded into the audio embedding chunk during tokenization.
            std::string mtmd_prompt = replace_all(
                    prompt,
                    "<|audio_start|><|audio_pad|><|audio_end|>",
                    mtmd_default_marker());
            if (mtmd_prompt.find(mtmd_default_marker()) == std::string::npos) {
                throw std::invalid_argument("prompt must contain one Qwen3-ASR audio marker");
            }

            bitmap_ptr bitmap(mtmd_bitmap_init_from_audio(
                    audio_size, audio_data));
            if (!bitmap) {
                throw std::runtime_error("failed to create mtmd audio bitmap");
            }

            mtmd_input_text text{
                mtmd_prompt.data(),
                mtmd_prompt.size(),
                true,  // add_special
                true,  // parse_special
            };
            const mtmd_bitmap * bitmaps[] = {bitmap.get()};
            chunks_ptr chunks(mtmd_input_chunks_init());
            const int32_t tokenize_result = mtmd_tokenize(
                    mtmd_.get(), chunks.get(), &text, bitmaps, 1);
            if (tokenize_result != 0) {
                throw std::runtime_error("mtmd_tokenize failed with code " + std::to_string(tokenize_result));
            }

            llama_pos n_past = 0;
            const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
            chunk_count = n_chunks;
            size_t media_chunk_index = 0;
            prompt_token_count = mtmd_helper_get_n_tokens(chunks.get());
            prompt_position_count = mtmd_helper_get_n_pos(chunks.get());
            if (prompt_position_count + 1 >= llama_n_ctx(context_.get()))
                throw std::runtime_error("CONTEXT_LIMIT: audio/context exceeds native context capacity");
            for (size_t i = 0; i < n_chunks; ++i) {
                if (cancel && cancel(nullptr)) throw std::runtime_error("ASR_CANCELLED");
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), i);
                const bool is_last = i + 1 == n_chunks;
                llama_pos new_n_past = n_past;
                int32_t result = 0;

                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t n_text_tokens = 0;
                    const llama_token * text_tokens = mtmd_input_chunk_get_tokens_text(
                            chunk, &n_text_tokens);
                    if (text_tokens && n_text_tokens > 0) {
                        prompt_text_tokens.insert(
                                prompt_text_tokens.end(), text_tokens, text_tokens + n_text_tokens);
                    }
                    result = mtmd_helper_eval_chunk_single(
                        mtmd_.get(), context_.get(), chunk, n_past, 0, n_batch_,
                            is_last, &new_n_past);
                } else {
                    // Audio and image chunks use different internal token
                    // structures. The generic chunk accessors work for both
                    // and correctly report the media span.
                    media_token_counts.push_back(mtmd_input_chunk_get_n_tokens(chunk));
                    media_position_counts.push_back(mtmd_input_chunk_get_n_pos(chunk));
                    result = mtmd_encode_chunk(mtmd_.get(), chunk);
                    if (result == 0) {
                        float * embedding = mtmd_get_output_embd(mtmd_.get());
                        if (!embedding) {
                            throw std::runtime_error("mtmd returned no audio embedding");
                        }
                        debug_embedding(
                                embedding,
                                mtmd_input_chunk_get_n_tokens(chunk),
                                llama_model_n_embd_inp(model_.get()),
                                media_chunk_index++);
                        if (embedding_dump_path && embedding_dim > 0) {
                            const size_t n_values = static_cast<size_t>(mtmd_input_chunk_get_n_tokens(chunk))
                                * static_cast<size_t>(embedding_dim);
                            all_embeddings.insert(all_embeddings.end(), embedding, embedding + n_values);
                        }
                        std::vector<float> embedding_bf16;
                        float * embedding_for_decode = embedding;
                        if (embed_bf16_enabled()) {
                            const size_t n_values = static_cast<size_t>(mtmd_input_chunk_get_n_tokens(chunk))
                                * static_cast<size_t>(llama_model_n_embd_inp(model_.get()));
                            embedding_bf16.resize(n_values);
                            for (size_t j = 0; j < n_values; ++j) {
                                embedding_bf16[j] = round_to_bf16(embedding[j]);
                            }
                            embedding_for_decode = embedding_bf16.data();
                        }
                        if (decode_embedding_dump_path && embedding_dim > 0) {
                            const size_t n_values = static_cast<size_t>(mtmd_input_chunk_get_n_tokens(chunk))
                                * static_cast<size_t>(embedding_dim);
                            all_decode_embeddings.insert(
                                all_decode_embeddings.end(), embedding_for_decode, embedding_for_decode + n_values);
                        }
                        result = mtmd_helper_decode_image_chunk(
                                mtmd_.get(), context_.get(), chunk, embedding_for_decode,
                                n_past, 0, n_batch_, &new_n_past, nullptr, nullptr);
                    }
                }
                if (result != 0) {
                    throw std::runtime_error("failed to evaluate mtmd chunk " + std::to_string(i) +
                                             ", code " + std::to_string(result));
                }
                n_past = new_n_past;
            }

            if (embedding_dump_path && embedding_dim > 0 && !all_embeddings.empty()) {
                dump_f32_matrix(
                    embedding_dump_path,
                    all_embeddings.data(),
                    (int64_t)(all_embeddings.size() / (size_t)embedding_dim),
                    embedding_dim);
            }
            if (decode_embedding_dump_path && embedding_dim > 0 && !all_decode_embeddings.empty()) {
                dump_f32_matrix(
                    decode_embedding_dump_path,
                    all_decode_embeddings.data(),
                    (int64_t)(all_decode_embeddings.size() / (size_t)embedding_dim),
                    embedding_dim);
            }

            sampler_ptr sampler(llama_sampler_init_greedy());
            if (!sampler) {
                throw std::runtime_error("failed to create greedy sampler");
            }

            const llama_vocab * vocab = llama_model_get_vocab(model_.get());
            first_logits_topk = debug_first_logits(context_.get(), vocab);
            generated.reserve(static_cast<size_t>(max_tokens));
            finish_reason = "length";

            for (int i = 0; i < max_tokens; ++i) {
                if (cancel && cancel(nullptr)) throw std::runtime_error("ASR_CANCELLED");
                if (n_past + 1 >= llama_n_ctx(context_.get())) break;
                const llama_token token = llama_sampler_sample(sampler.get(), context_.get(), -1);
                generated.push_back(token);
                llama_sampler_accept(sampler.get(), token);

                if (llama_vocab_is_eog(vocab, token)) {
                    finish_reason = "stop";
                    break;
                }

                llama_token next_token = token;
                llama_batch batch = llama_batch_get_one(&next_token, 1);
                if (llama_decode(context_.get(), batch) != 0) {
                    throw std::runtime_error("llama_decode failed during generation");
                }
                ++n_past;
            }
            // Decode the full token sequence: individual pieces can split a
            // UTF-8 character and llama_token_to_piece would corrupt it.
            const size_t text_count = finish_reason == "stop" ? generated.size() - 1 : generated.size();
            generated_text = detokenize_tokens(
                    std::vector<llama_token>(generated.begin(), generated.begin() + text_count), true);
        }

        return {generated_text, finish_reason == "length"};
    }
private:
    model_ptr model_;
    context_ptr context_;
    mtmd_ptr mtmd_;
    int n_batch_ = 1024;
};
thread_local std::string last_error;
thread_local std::string last_text;
thread_local bool last_truncated = false;
} // namespace

#ifdef _WIN32
#define R2T2_EXPORT extern "C" __declspec(dllexport)
#else
#define R2T2_EXPORT extern "C" __attribute__((visibility("default")))
#endif
R2T2_EXPORT int r2t2_api_version() { return 1; }
R2T2_EXPORT const char * r2t2_last_error() { return last_error.c_str(); }
R2T2_EXPORT bool r2t2_last_truncated() { return last_truncated; }
R2T2_EXPORT void * r2t2_create(const char * model, const char * projector,
        int n_ctx, int n_batch, int n_threads, int gpu_layers) {
    last_error.clear();
    try { return new Qwen3ASRNative(model, projector, n_ctx, n_batch, n_threads, gpu_layers != 0, gpu_layers); }
    catch (const std::exception & e) { last_error = e.what(); return nullptr; }
    catch (...) { last_error = "Unknown native model load error"; return nullptr; }
}
R2T2_EXPORT void r2t2_destroy(void * engine) { delete static_cast<Qwen3ASRNative *>(engine); }
R2T2_EXPORT const char * r2t2_generate(void * engine, const float * audio, size_t count,
        const char * prompt, int max_tokens, cancel_callback cancel) {
    last_error.clear(); last_text.clear(); last_truncated = false;
    try {
        if (!engine) throw std::invalid_argument("Native engine is not loaded");
        auto result = static_cast<Qwen3ASRNative *>(engine)->generate_once(audio, count, cancel, prompt, max_tokens);
        last_text = std::move(result.text); last_truncated = result.truncated;
        return last_text.c_str();
    } catch (const std::exception & e) { last_error = e.what(); return nullptr; }
      catch (...) { last_error = "Unknown native decode error"; return nullptr; }
}
