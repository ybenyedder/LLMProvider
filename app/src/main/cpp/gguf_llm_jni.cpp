// JNI bridge: vendored llama.cpp (v0.5.0) <-> com.tree4five.gguf.LlmNative.
//
// One LlmEngine per loaded model file. Each engine owns:
//   - ctx_gen : pooling NONE, used for text generation (tokens in) and for
//               generation from injected embeddings (soft prompt).
//   - ctx_embd: pooling MEAN, created on demand (embedText mode 1 only).
// token_embd.weight is read directly from a private mmap of the model file
// (located through the GGUF metadata, no_alloc) and dequantized row by row,
// so input-embedding lookup and mean-of-rows embedText (mode 0) never need a
// forward pass. v0.5.0 removed llama_get_model_tensor, and reading the file
// bytes is the right replacement anyway: they hold the un-repacked layout the
// dequantizer expects, while the CPU backend may repack its own copies.
//
// v0.5.0 API notes baked into this code:
//   - The legacy sampler API (llama_sample_temp/top_k/top_p/token[_greedy]) is
//     gone: sampling is hand-rolled (damp_repeats + greedy argmax, or
//     top-k/top-p/temperature softmax).
//   - llama_batch_get_one lost its pos_0 argument and auto-tracks positions;
//     we do not rely on that — make_token_batch builds batches with explicit
//     positions and "logits on last position only".
//   - A llama_batch must be either token-only or embd-only; the latent prefix
//     and the text follow-up are therefore two llama_decode calls with
//     continuous positions.
//   - llama_batch_init leaves n_tokens, pos/n_seq_id/seq_id/logits
//     UNINITIALIZED (zeroed): every field must be written before decode.
//   - llama_kv_cache_clear is now llama_memory_clear(llama_get_memory(ctx)).
//   - Vocabulary calls take a llama_vocab (llama_model_get_vocab), and the
//     non-deprecated accessors are llama_vocab_eos / llama_vocab_n_tokens /
//     llama_model_n_embd / llama_model_n_ctx_train.
//   - Repetition damping is still hand-rolled: greedy decoding (the service
//     constant temperature) cannot escape a repetition loop once a token
//     starts feeding back on itself — qwen2.5-0.5b q4 on arm64 emitted 512×
//     '!' while the x86_64 build escaped by luckier numerics (fixed for real
//     by bumping the vendored llama.cpp, see llama.cpp/VENDORED.md).
//
// JNI pitfalls handled here:
//   - JavaVM cached in JNI_OnLoad; GetEnv + attach for foreign native threads.
//   - GetObjectClass (never FindClass) to resolve the callback object class.
//   - Callback held as a NewGlobalRef for the duration of generation.
//   - ExceptionCheck after every Java callback, DeleteLocalRef per jstring.

#include <jni.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <android/log.h>

#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "ggufllm", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ggufllm", __VA_ARGS__)

namespace {

constexpr int kMaxBatch = 512;

struct LlmEngine {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_context * ctx_gen = nullptr;
    llama_context * ctx_embd = nullptr;   // on demand, pooling MEAN

    int n_embd = 0;
    int n_vocab = 0;
    int n_ctx = 0;

    // token_embd.weight, kept as raw (possibly quantized) rows for cheap
    // per-row dequantization without copying the whole table. Backed by a
    // private mmap of the model file (unmap in freeModel/destructor).
    void * tok_embd_map = nullptr;
    size_t tok_embd_map_size = 0;
    const void * tok_embd_data = nullptr;
    enum ggml_type tok_embd_type = GGML_TYPE_F32;
    size_t tok_embd_row_bytes = 0;

    std::atomic<bool> stop{false};
    // llama_context is not thread-safe; serialize everything on this engine.
    std::mutex mutex;

    ~LlmEngine() {
        if (ctx_embd) { llama_free(ctx_embd); }
        if (ctx_gen) { llama_free(ctx_gen); }
        if (model) { llama_model_free(model); }
        if (tok_embd_map) { munmap(tok_embd_map, tok_embd_map_size); }
    }
};

JavaVM * g_vm = nullptr;

JNIEnv * ensure_env() {
    JNIEnv * env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
        return env;
    }
    if (g_vm->AttachCurrentThreadAsDaemon(&env, nullptr) == JNI_OK) {
        return env;
    }
    return nullptr;
}

void throw_java(JNIEnv * env, const char * cls, const char * msg) {
    if (env->ExceptionCheck() == JNI_TRUE) { return; }  // a pending exception wins
    jclass c = env->FindClass(cls);
    if (c != nullptr) {
        env->ThrowNew(c, msg);
        env->DeleteLocalRef(c);
    }
}

std::string to_std_string(JNIEnv * env, jstring s) {
    if (s == nullptr) { return std::string(); }
    const char * chars = env->GetStringUTFChars(s, nullptr);
    if (chars == nullptr) { return std::string(); }
    std::string out(chars);
    env->ReleaseStringUTFChars(s, chars);
    return out;
}

LlmEngine * engine_of(JNIEnv * env, jlong handle) {
    auto * engine = reinterpret_cast<LlmEngine *>(static_cast<uintptr_t>(handle));
    if (engine == nullptr) {
        throw_java(env, "java/lang/IllegalArgumentException", "invalid model handle");
    }
    return engine;
}

// RAII lock over the engine mutex (llama_context is not thread-safe).
struct EngineLock {
    explicit EngineLock(LlmEngine * engine) : e(engine) { e->mutex.lock(); }
    ~EngineLock() { e->mutex.unlock(); }
    LlmEngine * e;
};

// ---------------------------------------------------------------- callbacks

struct CallbackRef {
    JNIEnv * env = nullptr;
    jobject obj = nullptr;       // global ref
    jclass cls = nullptr;        // global ref
    jmethodID on_token = nullptr;
    jmethodID on_error = nullptr;
    bool broken = false;         // a Java exception is pending: stop feeding

    bool init(JNIEnv * jni_env, jobject callback) {
        env = jni_env;
        if (callback == nullptr) { return false; }
        obj = env->NewGlobalRef(callback);
        cls = static_cast<jclass>(env->NewGlobalRef(env->GetObjectClass(callback)));
        on_token = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
        on_error = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
        return on_token != nullptr;
    }

    void token(const std::string & piece) {
        if (broken || on_token == nullptr) { return; }
        jstring s = env->NewStringUTF(piece.c_str());
        if (s == nullptr) { broken = true; return; }   // OOM
        env->CallVoidMethod(obj, on_token, s);
        env->DeleteLocalRef(s);
        if (env->ExceptionCheck() == JNI_TRUE) {
            broken = true;
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    }

    void error(const std::string & message) {
        if (broken || on_error == nullptr) { return; }
        jstring s = env->NewStringUTF(message.c_str());
        if (s == nullptr) { broken = true; return; }
        env->CallVoidMethod(obj, on_error, s);
        env->DeleteLocalRef(s);
        if (env->ExceptionCheck() == JNI_TRUE) {
            broken = true;
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    }

    void release(JNIEnv * jni_env) {
        if (obj) { jni_env->DeleteGlobalRef(obj); obj = nullptr; }
        if (cls) { jni_env->DeleteGlobalRef(cls); cls = nullptr; }
    }
};

// ---------------------------------------------------------------- sampling

// Reads the logits of the last output position and samples one token.
// llama.cpp throws C++ exceptions on invalid access; none may cross the JNI
// boundary (std::terminate), so every risky read is guarded here.
// Repetition damping (the legacy llama_sample_repetition_penalty is gone): a
// token seen in the recent window gets its logit pushed down, so decoding can
// no longer lock into a loop.
static constexpr float kRepeatPenalty = 1.10f;   // 1.0 = disabled
static constexpr int kPenaltyLastN = 64;

static void damp_repeats(float * logits, const int n_vocab,
                         const std::vector<llama_token> & recent) {
    if (kRepeatPenalty == 1.0f || recent.empty()) { return; }
    const int n = std::min<int>(kPenaltyLastN, static_cast<int>(recent.size()));
    std::set<llama_token> seen(recent.end() - n, recent.end());
    for (const llama_token t : seen) {
        if (t < 0 || t >= n_vocab) { continue; }
        logits[t] = (logits[t] <= 0.0f) ? logits[t] * kRepeatPenalty
                                        : logits[t] / kRepeatPenalty;
    }
}

llama_token sample_next(LlmEngine * e, llama_context * ctx, float temperature, int top_k,
                        float top_p, const std::vector<llama_token> & recent) {
    float * logits = nullptr;
    try {
        logits = llama_get_logits_ith(ctx, -1);
    } catch (const std::exception & ex) {
        fprintf(stderr, "ggufllm: sample_next failed: %s\n", ex.what());
        return llama_vocab_eos(e->vocab);
    }
    if (logits == nullptr) { return llama_vocab_eos(e->vocab); }
    const int n_vocab = e->n_vocab;
    damp_repeats(logits, n_vocab, recent);

    if (temperature <= 0.0f) {
        // Greedy: first index holding the strictly highest logit (ties keep
        // the lowest id, like the old llama_sample_token_greedy).
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) { best = i; }
        }
        return best;
    }

    // Candidate indices, sorted by descending logit.
    std::vector<int> idx(static_cast<size_t>(n_vocab));
    for (int i = 0; i < n_vocab; ++i) { idx[static_cast<size_t>(i)] = i; }
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[a] > logits[b]; });
    if (top_k > 0 && top_k < n_vocab) { idx.resize(static_cast<size_t>(top_k)); }

    if (top_p < 1.0f && !idx.empty()) {
        // Nucleus: smallest prefix of the sorted list whose softmax mass
        // reaches top_p. Softmax with the max subtracted for stability.
        const float top_logit = logits[idx[0]];
        std::vector<float> p(idx.size());
        float total = 0.0f;
        for (size_t i = 0; i < idx.size(); ++i) {
            p[i] = std::exp(logits[idx[i]] - top_logit);
            total += p[i];
        }
        float cum = 0.0f;
        size_t cut = idx.size();
        for (size_t i = 0; i < idx.size(); ++i) {
            cum += p[i] / total;
            if (cum >= top_p) { cut = i + 1; break; }
        }
        idx.resize(cut);
    }
    if (idx.empty()) { return llama_vocab_eos(e->vocab); }

    // Softmax over the survivors at the given temperature, then one draw.
    const float top_logit = logits[idx[0]];
    std::vector<double> p(idx.size());
    double total = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        p[i] = std::exp(static_cast<double>(logits[idx[i]] - top_logit) / temperature);
        total += p[i];
    }
    if (!(total > 0.0)) { return idx[0]; }
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> uni(0.0, total);
    const double r = uni(rng);
    double cum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        cum += p[i];
        if (r < cum) { return idx[i]; }
    }
    return idx.back();
}

// ---------------------------------------------------------------- batching

// One batch with explicit positions and "logits on last position only" (what
// the old llama_batch_get_one(tokens, n, pos_0, seq_id) with logits==nullptr
// meant). The v0.5.0 helper lost pos_0 and auto-tracks positions; the
// latent-prefix path needs explicit control, so batches are built here.
llama_batch make_token_batch(const llama_token * tokens, int n, int pos_0) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;  // llama_batch_init leaves this at 0!
    for (int i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos_0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1) ? 1 : 0;
    }
    return batch;
}

// ---------------------------------------------------------------- embedding

// Dequantize row `index` of token_embd into out (n_embd floats).
bool dequant_row(LlmEngine * e, int index, float * out) {
    const ggml_type_traits * t = ggml_get_type_traits(e->tok_embd_type);
    if (t == nullptr || t->to_float == nullptr) { return false; }
    const uint8_t * row = static_cast<const uint8_t *>(e->tok_embd_data) +
                          static_cast<size_t>(index) * e->tok_embd_row_bytes;
    t->to_float(row, out, e->n_embd);
    return true;
}

// Map the raw token_embd rows from the model file. `offset` is the absolute
// file offset of the tensor data (data section offset + tensor offset from
// the GGUF metadata); rows are then row_bytes apart from that pointer.
bool map_tok_embd(LlmEngine * e, const std::string & file, size_t offset) {
    const int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) { return false; }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); return false; }
    // mmap only accepts page-aligned file offsets: map from the enclosing page.
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t map_off = offset & ~(page - 1);
    const size_t rel = offset - map_off;
    const size_t len = static_cast<size_t>(st.st_size) - map_off;
    void * p = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd,
                    static_cast<off_t>(map_off));
    close(fd);
    if (p == MAP_FAILED) { return false; }
    e->tok_embd_map = p;
    e->tok_embd_map_size = len;
    e->tok_embd_data = static_cast<const uint8_t *>(p) + rel;
    return true;
}

// --------------------------------------------------------------- tokenizing

// Tokenize into tokens; returns the number written, 0 on empty, -1 on failure.
int tokenize_to(LlmEngine * e, const std::string & s, std::vector<llama_token> & tokens,
                bool add_special) {
    int n = llama_tokenize(e->vocab, s.data(), static_cast<int>(s.size()), nullptr, 0,
                           add_special, true);
    if (n < 0) { n = -n; }
    if (n <= 0) { return 0; }
    tokens.resize(static_cast<size_t>(n));
    const int written = llama_tokenize(e->vocab, s.data(), static_cast<int>(s.size()),
                                       tokens.data(), n, add_special, true);
    if (written <= 0) { return -1; }
    tokens.resize(static_cast<size_t>(written));
    return written;
}

// --------------------------------------------------------------- generation

// llama_decode throws C++ exceptions on invalid input (e.g. batch overflows
// n_ctx); none may cross the JNI boundary.
int safe_decode(llama_context * ctx, llama_batch & batch) {
    try {
        return llama_decode(ctx, batch);
    } catch (const std::exception & ex) {
        ALOGE("llama_decode failed: %s", ex.what());
        return -1;
    } catch (...) {
        ALOGE("llama_decode failed: unknown error");
        return -1;
    }
}

// Append piece(tok) to out; returns false when tok is EOS.
bool append_piece(LlmEngine * e, llama_token tok, std::string & out) {
    if (tok == llama_vocab_eos(e->vocab)) { return false; }
    std::vector<char> piece(256);
    int n = llama_token_to_piece(e->vocab, tok, piece.data(), static_cast<int>(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(e->vocab, tok, piece.data(), static_cast<int>(piece.size()), 0, true);
    }
    if (n > 0) { out.append(piece.data(), static_cast<size_t>(n)); }
    return true;
}

// Sampling loop shared by both generation entry points. `pos` is the next KV
// position; every sampled token is decoded at pos++ (continuous positions).
// `prompt` seeds the repetition window (latent-prefill paths pass the text
// follow-up tokens; the latent prefix itself carries no token ids).
int run_sampling(LlmEngine * e, llama_context * ctx, JNIEnv * env, jobject callback,
                 int n_predict, float temperature, int top_k, float top_p, int pos,
                 const std::vector<llama_token> & prompt) {
    CallbackRef cb;
    const bool has_cb = cb.init(env, callback);
    std::vector<llama_token> recent(prompt);
    int produced = 0;
    while (produced < n_predict && !e->stop.load()) {
        const llama_token tok = sample_next(e, ctx, temperature, top_k, top_p, recent);
        if (produced == 0) {
            ALOGI("sampling: first token=%d (eos=%d)", tok, llama_vocab_eos(e->vocab));
        }
        std::string piece;
        if (!append_piece(e, tok, piece)) { break; }              // EOS
        if (!piece.empty() && has_cb) {
            cb.token(piece);
            if (cb.broken) { break; }
        }
        llama_batch batch = make_token_batch(&tok, 1, pos);
        const int rc = safe_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) { break; }
        recent.push_back(tok);
        if (recent.size() > 4096) { recent.erase(recent.begin(), recent.begin() + 2048); }
        ++pos;
        ++produced;
    }
    cb.release(env);
    return produced;
}

bool build_ctx_gen(LlmEngine * e, int n_ctx, int n_threads) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(n_ctx);
    cp.n_batch = kMaxBatch;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.embeddings = false;
    e->ctx_gen = llama_init_from_model(e->model, cp);
    return e->ctx_gen != nullptr;
}

}  // namespace

// ================================================================== exported

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM * vm, void * /*reserved*/) {
    g_vm = vm;
    JNIEnv * env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL
Java_com_tree4five_gguf_LlmNative_loadModel(JNIEnv * env, jobject /*thiz*/,
                                            jstring path, jint n_ctx, jint n_threads) {
    const std::string file = to_std_string(env, path);
    auto * e = new (std::nothrow) LlmEngine();
    if (e == nullptr) {
        throw_java(env, "java/lang/OutOfMemoryError", "cannot allocate engine");
        return 0;
    }
    // Hash the file we are about to load with a sequential read (exercises
    // the storage as well as the bytes). GGUF carries no content checksum, so
    // a silently corrupted copy loads fine and then poisons the forward pass
    // with NaN logits — this line in the log is how such copies are caught
    // (compare the value across devices carrying the same model). FNV-1a 64:
    // a fingerprint, never an integrity guarantee.
    {
        const int fd = open(file.c_str(), O_RDONLY);
        if (fd >= 0) {
            uint64_t h = 1469598103934665603ULL;
            long long total = 0;
            std::vector<char> buf(4 << 20);
            ssize_t n;
            while ((n = read(fd, buf.data(), buf.size())) > 0) {
                for (ssize_t i = 0; i < n; ++i) {
                    h = (h ^ static_cast<uint8_t>(buf[static_cast<size_t>(i)])) * 1099511628211ULL;
                }
                total += n;
            }
            close(fd);
            ALOGI("loadModel: file fnv1a=%08x%08x size=%lld",
                  static_cast<uint32_t>(h >> 32), static_cast<uint32_t>(h), total);
        }
    }

    llama_model_params mp = llama_model_default_params();
    // CPU-only: the target tablet has no usable GPU offload; mmap keeps RSS low.
    mp.n_gpu_layers = 0;
    e->model = llama_model_load_from_file(file.c_str(), mp);
    if (e->model == nullptr) {
        delete e;
        throw_java(env, "java/io/IOException", "failed to load model file");
        return 0;
    }
    e->vocab = llama_model_get_vocab(e->model);
    e->n_embd = llama_model_n_embd(e->model);
    e->n_vocab = e->vocab != nullptr ? llama_vocab_n_tokens(e->vocab) : 0;
    e->n_ctx = n_ctx > 0 ? n_ctx : 2048;

    // Cache the token embedding table for input-embedding lookup. v0.5.0
    // removed llama_get_model_tensor, so locate the tensor through the GGUF
    // metadata (no_alloc) and keep a private mmap of the file bytes.
    if (e->vocab != nullptr) {
        gguf_init_params gp{};
        gp.no_alloc = true;
        gp.ctx = nullptr;
        gguf_context * gguf = gguf_init_from_file(file.c_str(), gp);
        if (gguf != nullptr) {
            const int64_t idx = gguf_find_tensor(gguf, "token_embd.weight");
            if (idx >= 0) {
                const size_t offset = gguf_get_data_offset(gguf) +
                                      gguf_get_tensor_offset(gguf, idx);
                e->tok_embd_type = gguf_get_tensor_type(gguf, idx);
                e->tok_embd_row_bytes = ggml_row_size(e->tok_embd_type, e->n_embd);
                if (!map_tok_embd(e, file, offset)) {
                    e->tok_embd_data = nullptr;   // fall back to text handling
                }
            }
            gguf_free(gguf);
        }
    }
    ALOGI("loadModel: n_embd=%d n_vocab=%d token_embd=%s", e->n_embd, e->n_vocab,
          e->tok_embd_data != nullptr ? "mapped" : "unavailable");

    if (!build_ctx_gen(e, e->n_ctx, n_threads > 0 ? n_threads : 2)) {
        llama_model_free(e->model);
        delete e;
        throw_java(env, "java/io/IOException", "failed to create llama context");
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(e));
}

JNIEXPORT void JNICALL
Java_com_tree4five_gguf_LlmNative_freeModel(JNIEnv * env, jobject /*thiz*/, jlong handle) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return; }
    std::lock_guard<std::mutex> lock(e->mutex);
    if (e->ctx_embd) { llama_free(e->ctx_embd); e->ctx_embd = nullptr; }
    if (e->ctx_gen) { llama_free(e->ctx_gen); e->ctx_gen = nullptr; }
    if (e->model) { llama_model_free(e->model); e->model = nullptr; }
    if (e->tok_embd_map) { munmap(e->tok_embd_map, e->tok_embd_map_size); e->tok_embd_map = nullptr; }
    delete e;  // safe: the caller must never reuse the handle after freeModel
}

JNIEXPORT jint JNICALL
Java_com_tree4five_gguf_LlmNative_getEmbeddingDim(JNIEnv * env, jobject /*thiz*/, jlong handle) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return -1; }
    // -1 when the embedding table could not be located: the caller falls back
    // to text-based context handling.
    return e->tok_embd_data != nullptr ? e->n_embd : -1;
}

JNIEXPORT jint JNICALL
Java_com_tree4five_gguf_LlmNative_getContextLength(JNIEnv * env, jobject /*thiz*/, jlong handle) {
    auto * e = engine_of(env, handle);
    if (e == nullptr || e->model == nullptr) { return -1; }
    // Value straight from the GGUF metadata (train context window), capped by
    // the runtime window this engine actually allocates.
    int train = static_cast<int>(llama_model_n_ctx_train(e->model));
    return train > 0 ? std::min(train, e->n_ctx) : e->n_ctx;
}

JNIEXPORT jintArray JNICALL
Java_com_tree4five_gguf_LlmNative_tokenize(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                           jstring text, jboolean add_special, jboolean parse_special) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return nullptr; }
    const std::string s = to_std_string(env, text);
    const bool add_special_b = add_special == JNI_TRUE;
    int n = llama_tokenize(e->vocab, s.data(), static_cast<int>(s.size()), nullptr, 0,
                           add_special_b, parse_special == JNI_TRUE);
    if (n < 0) { n = -n; }
    if (n <= 0) { return env->NewIntArray(0); }
    std::vector<llama_token> tokens(static_cast<size_t>(n));
    const int written = llama_tokenize(e->vocab, s.data(), static_cast<int>(s.size()),
                                       tokens.data(), n, add_special_b, parse_special == JNI_TRUE);
    if (written < 0) {
        throw_java(env, "java/lang/IllegalStateException", "tokenization failed");
        return nullptr;
    }
    jintArray out = env->NewIntArray(written);
    if (out == nullptr) { return nullptr; }
    env->SetIntArrayRegion(out, 0, written, reinterpret_cast<const jint *>(tokens.data()));
    return out;
}

JNIEXPORT jstring JNICALL
Java_com_tree4five_gguf_LlmNative_tokenToPiece(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                               jint token) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return nullptr; }
    std::vector<char> piece(256);
    int n = llama_token_to_piece(e->vocab, token, piece.data(), static_cast<int>(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(e->vocab, token, piece.data(), static_cast<int>(piece.size()), 0, true);
    }
    if (n <= 0) { return env->NewStringUTF(""); }
    return env->NewStringUTF(std::string(piece.data(), static_cast<size_t>(n)).c_str());
}

// Input-embedding lookup: returns a flat array of len(tokens) * n_embd floats
// holding the dequantized token_embd row of every id (row t at [t*n_embd,
// (t+1)*n_embd)). This is exactly what a token-input llama_decode gathers, so
// feeding these rows to generateFromEmbeddings reproduces the text path.
JNIEXPORT jfloatArray JNICALL
Java_com_tree4five_gguf_LlmNative_embedTokenRows(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                                 jintArray tokens) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return nullptr; }
    if (e->tok_embd_data == nullptr) {
        throw_java(env, "java/lang/IllegalStateException", "model has no accessible token_embd");
        return nullptr;
    }
    const jsize n = env->GetArrayLength(tokens);
    if (n <= 0) { return env->NewFloatArray(0); }
    std::vector<jint> ids(static_cast<size_t>(n));
    env->GetIntArrayRegion(tokens, 0, n, ids.data());

    jfloatArray out = env->NewFloatArray(n * e->n_embd);
    if (out == nullptr) { return nullptr; }
    std::vector<float> row(static_cast<size_t>(e->n_embd));
    for (jsize t = 0; t < n; ++t) {
        const jint id = ids[static_cast<size_t>(t)];
        if (id < 0 || id >= e->n_vocab) { continue; }
        if (!dequant_row(e, id, row.data())) {
            env->DeleteLocalRef(out);
            throw_java(env, "java/lang/IllegalStateException", "unsupported token_embd type");
            return nullptr;
        }
        env->SetFloatArrayRegion(out, static_cast<jsize>(t) * e->n_embd, e->n_embd, row.data());
    }
    return out;
}

// mode 0 = mean of token-embedding rows (no forward pass, microseconds).
// mode 1 = mean of hidden states (pooling MEAN forward pass, seconds).
JNIEXPORT jfloatArray JNICALL
Java_com_tree4five_gguf_LlmNative_embedText(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                            jstring text, jint mode) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return nullptr; }
    EngineLock lock(e);
    const std::string s = to_std_string(env, text);
    std::vector<llama_token> tokens;
    if (tokenize_to(e, s, tokens, /*add_special=*/true) <= 0) { return env->NewFloatArray(0); }
    if (static_cast<int>(tokens.size()) > kMaxBatch) {
        tokens.resize(static_cast<size_t>(kMaxBatch));  // single-decode path
    }

    if (mode == 1) {
        if (e->ctx_embd == nullptr) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = kMaxBatch;
            cp.n_batch = kMaxBatch;
            cp.n_threads = 2;
            cp.n_threads_batch = 2;
            cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
            cp.embeddings = true;
            e->ctx_embd = llama_init_from_model(e->model, cp);
            if (e->ctx_embd == nullptr) {
                throw_java(env, "java/lang/IllegalStateException", "failed to create embedding context");
                return nullptr;
            }
        }
        llama_batch batch = make_token_batch(tokens.data(), static_cast<int>(tokens.size()), 0);
        const int rc = safe_decode(e->ctx_embd, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw_java(env, "java/lang/IllegalStateException", "embedding decode failed");
            return nullptr;
        }
        // With pooling MEAN the pooled vector lives in embd_seq, not in the
        // per-token embedding buffer.
        const float * embd = llama_get_embeddings_seq(e->ctx_embd, 0);
        if (embd == nullptr) {
            embd = llama_get_embeddings(e->ctx_embd);
        }
        if (embd == nullptr) {
            throw_java(env, "java/lang/IllegalStateException", "no embeddings returned");
            return nullptr;
        }
        jfloatArray out = env->NewFloatArray(e->n_embd);
        if (out == nullptr) { return nullptr; }
        env->SetFloatArrayRegion(out, 0, e->n_embd, embd);
        return out;
    }

    if (e->tok_embd_data == nullptr) {
        throw_java(env, "java/lang/IllegalStateException", "model has no accessible token_embd");
        return nullptr;
    }
    std::vector<float> row(static_cast<size_t>(e->n_embd));
    std::vector<double> acc(static_cast<size_t>(e->n_embd), 0.0);
    int used = 0;
    for (const llama_token id : tokens) {
        if (id < 0 || id >= e->n_vocab) { continue; }
        if (!dequant_row(e, id, row.data())) {
            throw_java(env, "java/lang/IllegalStateException", "unsupported token_embd type");
            return nullptr;
        }
        for (int d = 0; d < e->n_embd; ++d) { acc[static_cast<size_t>(d)] += row[static_cast<size_t>(d)]; }
        ++used;
    }
    if (used == 0) { return env->NewFloatArray(0); }
    std::vector<float> mean(static_cast<size_t>(e->n_embd));
    for (int d = 0; d < e->n_embd; ++d) {
        mean[static_cast<size_t>(d)] = static_cast<float>(acc[static_cast<size_t>(d)] / used);
    }
    jfloatArray out = env->NewFloatArray(e->n_embd);
    if (out == nullptr) { return nullptr; }
    env->SetFloatArrayRegion(out, 0, e->n_embd, mean.data());
    return out;
}

// Generate from `prompt` (plain text path).
JNIEXPORT jint JNICALL
Java_com_tree4five_gguf_LlmNative_generate(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                           jstring prompt, jint n_predict, jfloat temperature,
                                           jint top_k, jfloat top_p, jobject callback) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return -1; }
    EngineLock lock(e);

    const std::string s = to_std_string(env, prompt);
    std::vector<llama_token> tokens;
    if (tokenize_to(e, s, tokens, /*add_special=*/true) <= 0) { return 0; }
    if (static_cast<int>(tokens.size()) > e->n_ctx - 4) {
        tokens.resize(static_cast<size_t>(e->n_ctx) - 4);
    }

    llama_context * ctx = e->ctx_gen;
    llama_memory_clear(llama_get_memory(ctx), true);
    e->stop.store(false);

    int pos = 0;
    while (pos < static_cast<int>(tokens.size())) {
        const int chunk = std::min(kMaxBatch, static_cast<int>(tokens.size()) - pos);
        llama_batch batch = make_token_batch(tokens.data() + pos, chunk, pos);
        const int rc = safe_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) { return -1; }
        pos += chunk;
    }
    // logits was set on the last position of every chunk, so -1 indexes the
    // logits the sampling loop wants.

    return run_sampling(e, ctx, env, callback, n_predict, temperature, top_k, top_p, pos,
                        tokens);
}

// Generate from injected embeddings: `vectors` is a flat float array holding
// count rows of n_embd floats (the soft prompt), optionally followed by a
// text follow-up prompt (ChatML-formatted by the Kotlin layer) before
// sampling starts.
JNIEXPORT jint JNICALL
Java_com_tree4five_gguf_LlmNative_generateFromEmbeddings(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                                         jfloatArray vectors, jint count,
                                                         jstring followup_prompt, jint n_predict,
                                                         jfloat temperature, jint top_k, jfloat top_p,
                                                         jobject callback) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return -1; }
    if (count <= 0) { return 0; }
    EngineLock lock(e);
    if (e->n_embd <= 0) { return -1; }

    const jsize expected = count * e->n_embd;
    if (env->GetArrayLength(vectors) < expected) {
        throw_java(env, "java/lang/IllegalArgumentException", "embedding array too small");
        return -1;
    }
    std::vector<float> flat(static_cast<size_t>(expected));
    env->GetFloatArrayRegion(vectors, 0, expected, flat.data());

    llama_context * ctx = e->ctx_gen;
    llama_memory_clear(llama_get_memory(ctx), true);
    e->stop.store(false);

    const std::string followup = to_std_string(env, followup_prompt);
    ALOGI("generateFromEmbeddings: count=%d n_embd=%d followup_chars=%zu", count, e->n_embd, followup.size());

    // 1) Prefill the latent prefix in chunks (batches must be embd-only; the
    //    text follow-up is a second decode with continuous positions). With no
    //    follow-up, the last prefix token must produce the sampling logits.
    const bool followup_empty = followup.empty();
    int pos = 0;
    while (pos < count) {
        const int chunk = std::min(kMaxBatch, count - pos);
        const bool is_last_chunk = (pos + chunk >= count);
        llama_batch batch = llama_batch_init(chunk, e->n_embd, 1);
        batch.n_tokens = chunk;  // llama_batch_init leaves this at 0!
        for (int i = 0; i < chunk; ++i) {
            const size_t src = static_cast<size_t>(pos + i) * e->n_embd;
            float * dst = batch.embd + static_cast<size_t>(i) * e->n_embd;
            for (int d = 0; d < e->n_embd; ++d) { dst[d] = flat[src + d]; }
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (followup_empty && is_last_chunk && i == chunk - 1) ? 1 : 0;
        }
        const int rc = safe_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) { ALOGE("latent prefill decode failed at pos=%d rc=%d", pos, rc); return -1; }
        pos += chunk;
    }
    ALOGI("latent prefill done: pos=%d followup_empty=%d", pos, followup_empty ? 1 : 0);

    // 2) Optional text follow-up.
    std::vector<llama_token> followup_tokens;
    if (!followup_empty) {
        if (tokenize_to(e, followup, followup_tokens, /*add_special=*/true) > 0) {
            if (static_cast<int>(followup_tokens.size()) > e->n_ctx - pos - 4) {
                followup_tokens.resize(
                    static_cast<size_t>(std::max(0, e->n_ctx - pos - 4)));
            }
            int p2 = 0;
            while (p2 < static_cast<int>(followup_tokens.size())) {
                const int chunk = std::min(kMaxBatch, static_cast<int>(followup_tokens.size()) - p2);
                llama_batch batch = make_token_batch(followup_tokens.data() + p2, chunk, pos);
                const int rc = safe_decode(ctx, batch);
                llama_batch_free(batch);
                if (rc != 0) { return -1; }
                pos += chunk;
                p2 += chunk;
            }
        }
    }

    // 3) Sampling loop (logits from the last decoded position).
    return run_sampling(e, ctx, env, callback, n_predict, temperature, top_k, top_p, pos,
                        followup_tokens);
}

JNIEXPORT void JNICALL
Java_com_tree4five_gguf_LlmNative_stop(JNIEnv * env, jobject /*thiz*/, jlong handle) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return; }
    e->stop.store(true);
}

}  // extern "C"
