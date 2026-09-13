// JNI bridge: vendored llama.cpp (b3621) <-> com.tree4five.gguf.LlmNative.
//
// One LlmEngine per loaded model file. Each engine owns:
//   - ctx_gen : pooling NONE, used for text generation (tokens in) and for
//               generation from injected embeddings (soft prompt).
//   - ctx_embd: pooling MEAN, created on demand (embedText mode 1 only).
// token_embd.weight is read directly from the mmap'd model buffer (CPU
// backend) and dequantized row by row, so input-embedding lookup and
// mean-of-rows embedText (mode 0) never need a forward pass.
//
// b3621 API notes baked into this code:
//   - llama_batch_get_one(tokens, n, pos_0, seq_id) leaves logits == nullptr,
//     which decode() treats as "output last position only"; the matching
//     logits are then llama_get_logits_ith(ctx, -1).
//   - A llama_batch must be either token-only or embd-only; the latent prefix
//     and the text follow-up are therefore two llama_decode calls with
//     continuous positions.
//   - llama_batch_init leaves n_tokens, pos/n_seq_id/seq_id/logits
//     UNINITIALIZED (zeroed): every field must be written before decode.
//   - Sampling is the legacy API (llama_sample_temp/top_k/top_p/token[_greedy]).
//
// JNI pitfalls handled here:
//   - JavaVM cached in JNI_OnLoad; GetEnv + attach for foreign native threads.
//   - GetObjectClass (never FindClass) to resolve the callback object class.
//   - Callback held as a NewGlobalRef for the duration of generation.
//   - ExceptionCheck after every Java callback, DeleteLocalRef per jstring.

#include <jni.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml.h"

#include <android/log.h>

#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "ggufllm", __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ggufllm", __VA_ARGS__)

namespace {

constexpr int kMaxBatch = 512;

struct LlmEngine {
    llama_model * model = nullptr;
    llama_context * ctx_gen = nullptr;
    llama_context * ctx_embd = nullptr;   // on demand, pooling MEAN

    int n_embd = 0;
    int n_vocab = 0;
    int n_ctx = 0;

    // token_embd.weight, kept as raw (possibly quantized) rows for cheap
    // per-row dequantization without copying the whole table.
    const void * tok_embd_data = nullptr;
    enum ggml_type tok_embd_type = GGML_TYPE_F32;
    size_t tok_embd_row_bytes = 0;

    std::atomic<bool> stop{false};
    // llama_context is not thread-safe; serialize everything on this engine.
    std::mutex mutex;

    ~LlmEngine() {
        if (ctx_embd) { llama_free(ctx_embd); }
        if (ctx_gen) { llama_free(ctx_gen); }
        if (model) { llama_free_model(model); }
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
llama_token sample_next(llama_context * ctx, float temperature, int top_k, float top_p) {
    const float * logits = nullptr;
    try {
        logits = llama_get_logits_ith(ctx, -1);
    } catch (const std::exception & ex) {
        fprintf(stderr, "ggufllm: sample_next failed: %s\n", ex.what());
        return llama_token_eos(llama_get_model(ctx));
    }
    if (logits == nullptr) { return llama_token_eos(llama_get_model(ctx)); }
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    std::vector<llama_token_data> candidates;
    candidates.reserve(static_cast<size_t>(n_vocab));
    for (int i = 0; i < n_vocab; ++i) {
        candidates.emplace_back(llama_token_data{i, logits[i], 0.0f});
    }
    llama_token_data_array arr{candidates.data(), candidates.size(), false};

    if (temperature <= 0.0f) {
        return llama_sample_token_greedy(ctx, &arr);
    }
    if (top_k > 0) { llama_sample_top_k(ctx, &arr, top_k, 1); }
    if (top_p < 1.0f) { llama_sample_top_p(ctx, &arr, top_p, 1); }
    llama_sample_temp(ctx, &arr, temperature);
    return llama_sample_token(ctx, &arr);
}

// ---------------------------------------------------------------- embedding

// Dequantize row `index` of token_embd into out (n_embd floats).
bool dequant_row(LlmEngine * e, int index, float * out) {
    const ggml_type_traits_t t = ggml_internal_get_type_traits(e->tok_embd_type);
    if (t.to_float == nullptr) { return false; }
    const uint8_t * row = static_cast<const uint8_t *>(e->tok_embd_data) +
                          static_cast<size_t>(index) * e->tok_embd_row_bytes;
    t.to_float(row, out, e->n_embd);
    return true;
}

// --------------------------------------------------------------- tokenizing

// Tokenize into tokens; returns the number written, 0 on empty, -1 on failure.
int tokenize_to(LlmEngine * e, const std::string & s, std::vector<llama_token> & tokens,
                bool add_special) {
    int n = llama_tokenize(e->model, s.data(), static_cast<int>(s.size()), nullptr, 0,
                           add_special, true);
    if (n < 0) { n = -n; }
    if (n <= 0) { return 0; }
    tokens.resize(static_cast<size_t>(n));
    const int written = llama_tokenize(e->model, s.data(), static_cast<int>(s.size()),
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
    if (tok == llama_token_eos(e->model)) { return false; }
    std::vector<char> piece(256);
    int n = llama_token_to_piece(e->model, tok, piece.data(), static_cast<int>(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(e->model, tok, piece.data(), static_cast<int>(piece.size()), 0, true);
    }
    if (n > 0) { out.append(piece.data(), static_cast<size_t>(n)); }
    return true;
}

// Sampling loop shared by both generation entry points. `pos` is the next KV
// position; every sampled token is decoded at pos++ (continuous positions).
int run_sampling(LlmEngine * e, llama_context * ctx, JNIEnv * env, jobject callback,
                 int n_predict, float temperature, int top_k, float top_p, int pos) {
    CallbackRef cb;
    const bool has_cb = cb.init(env, callback);
    int produced = 0;
    while (produced < n_predict && !e->stop.load()) {
        const llama_token tok = sample_next(ctx, temperature, top_k, top_p);
        if (produced == 0) {
            ALOGI("sampling: first token=%d (eos=%d)", tok, llama_token_eos(e->model));
        }
        std::string piece;
        if (!append_piece(e, tok, piece)) { break; }              // EOS
        if (!piece.empty() && has_cb) {
            cb.token(piece);
            if (cb.broken) { break; }
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&tok), 1, pos, 0);
        if (safe_decode(ctx, batch) != 0) { break; }
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
    e->ctx_gen = llama_new_context_with_model(e->model, cp);
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
    llama_model_params mp = llama_model_default_params();
    // CPU-only: the target tablet has no usable GPU offload; mmap keeps RSS low.
    mp.n_gpu_layers = 0;
    e->model = llama_load_model_from_file(file.c_str(), mp);
    if (e->model == nullptr) {
        delete e;
        throw_java(env, "java/io/IOException", "failed to load model file");
        return 0;
    }
    e->n_embd = llama_n_embd(e->model);
    e->n_vocab = llama_n_vocab(e->model);
    e->n_ctx = n_ctx > 0 ? n_ctx : 2048;

    // Cache the token embedding table for input-embedding lookup.
    ggml_tensor * tok_embd = llama_get_model_tensor(e->model, "token_embd.weight");
    if (tok_embd != nullptr && tok_embd->data != nullptr) {
        e->tok_embd_data = tok_embd->data;
        e->tok_embd_type = tok_embd->type;
        e->tok_embd_row_bytes = ggml_row_size(tok_embd->type, tok_embd->ne[0]);
    }

    if (!build_ctx_gen(e, e->n_ctx, n_threads > 0 ? n_threads : 2)) {
        llama_free_model(e->model);
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
    if (e->model) { llama_free_model(e->model); e->model = nullptr; }
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
    int train = static_cast<int>(llama_n_ctx_train(e->model));
    return train > 0 ? std::min(train, e->n_ctx) : e->n_ctx;
}

JNIEXPORT jintArray JNICALL
Java_com_tree4five_gguf_LlmNative_tokenize(JNIEnv * env, jobject /*thiz*/, jlong handle,
                                           jstring text, jboolean add_special, jboolean parse_special) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return nullptr; }
    const std::string s = to_std_string(env, text);
    const bool add_special_b = add_special == JNI_TRUE;
    int n = llama_tokenize(e->model, s.data(), static_cast<int>(s.size()), nullptr, 0,
                           add_special_b, parse_special == JNI_TRUE);
    if (n < 0) { n = -n; }
    if (n <= 0) { return env->NewIntArray(0); }
    std::vector<llama_token> tokens(static_cast<size_t>(n));
    const int written = llama_tokenize(e->model, s.data(), static_cast<int>(s.size()),
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
    int n = llama_token_to_piece(e->model, token, piece.data(), static_cast<int>(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(e->model, token, piece.data(), static_cast<int>(piece.size()), 0, true);
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
            e->ctx_embd = llama_new_context_with_model(e->model, cp);
            if (e->ctx_embd == nullptr) {
                throw_java(env, "java/lang/IllegalStateException", "failed to create embedding context");
                return nullptr;
            }
        }
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int>(tokens.size()), 0, 0);
        if (safe_decode(e->ctx_embd, batch) != 0) {
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
    llama_kv_cache_clear(ctx);
    e->stop.store(false);

    int pos = 0;
    while (pos < static_cast<int>(tokens.size())) {
        const int chunk = std::min(kMaxBatch, static_cast<int>(tokens.size()) - pos);
        llama_batch batch = llama_batch_get_one(tokens.data() + pos, chunk, pos, 0);
        if (safe_decode(ctx, batch) != 0) { return -1; }
        pos += chunk;
    }
    // logits==nullptr meant "last position only", so -1 indexes that output.

    return run_sampling(e, ctx, env, callback, n_predict, temperature, top_k, top_p, pos);
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
    llama_kv_cache_clear(ctx);
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
    if (!followup_empty) {
        std::vector<llama_token> tokens;
        if (tokenize_to(e, followup, tokens, /*add_special=*/true) > 0) {
            if (static_cast<int>(tokens.size()) > e->n_ctx - pos - 4) {
                tokens.resize(static_cast<size_t>(std::max(0, e->n_ctx - pos - 4)));
            }
            int p2 = 0;
            while (p2 < static_cast<int>(tokens.size())) {
                const int chunk = std::min(kMaxBatch, static_cast<int>(tokens.size()) - p2);
                llama_batch batch = llama_batch_get_one(tokens.data() + p2, chunk, pos, 0);
                if (safe_decode(ctx, batch) != 0) { return -1; }
                pos += chunk;
                p2 += chunk;
            }
        }
    }

    // 3) Sampling loop (logits from the last decoded position).
    return run_sampling(e, ctx, env, callback, n_predict, temperature, top_k, top_p, pos);
}

JNIEXPORT void JNICALL
Java_com_tree4five_gguf_LlmNative_stop(JNIEnv * env, jobject /*thiz*/, jlong handle) {
    auto * e = engine_of(env, handle);
    if (e == nullptr) { return; }
    e->stop.store(true);
}

}  // extern "C"
