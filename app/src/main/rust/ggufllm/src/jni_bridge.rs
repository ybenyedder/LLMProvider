//! JNI bridge (`com.tree4five.gguf.LlmNative`) + Engine coordinator (prefix cache,
//! batched prefill, N-gram speculative verification, live telemetry).

use crate::gguf::Gguf;
use crate::jni_sys::*;
use crate::model::{Model, State};
use crate::ngram::NgramDrafter;
use crate::ops::Sampler;
use crate::simd;
use crate::tokenizer::Utf8StreamDecoder;
use std::ffi::{CString, c_char, c_void};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const K_PREFILL_CHUNK: usize = 32;
const K_SPEC_DRAFT_MAX: usize = 4;

#[cfg(target_os = "android")]
mod logcat {
    use std::ffi::{CString, c_char, c_int};
    unsafe extern "C" {
        fn __android_log_write(prio: c_int, tag: *const c_char, text: *const c_char) -> c_int;
    }
    pub fn info(msg: &str) {
        if let (Ok(tag), Ok(cmsg)) = (CString::new("ggufllm-rust"), CString::new(msg)) {
            unsafe {
                __android_log_write(4, tag.as_ptr(), cmsg.as_ptr());
            }
        }
    }
    pub fn err(msg: &str) {
        if let (Ok(tag), Ok(cmsg)) = (CString::new("ggufllm-rust"), CString::new(msg)) {
            unsafe {
                __android_log_write(6, tag.as_ptr(), cmsg.as_ptr());
            }
        }
    }
}

#[cfg(not(target_os = "android"))]
mod logcat {
    pub fn info(msg: &str) {
        eprintln!("[ggufllm-rust] INFO: {msg}");
    }
    pub fn err(msg: &str) {
        eprintln!("[ggufllm-rust] ERROR: {msg}");
    }
}

pub struct EngineMetrics {
    pub last_prefill_tok_s: f64,
    pub last_decode_tok_s: f64,
    pub total_tokens_generated: u64,
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    pub prefix_cache_hits: u64,
    pub fingerprint: u64,
    pub file_size_mb: f64,
}

pub struct RuntimeCfg {
    pub spec_decoding: bool,
    pub prefix_cache: bool,
    pub repeat_penalty: f32,
}

pub struct EngineInner {
    pub model: Model<'static>,
    pub state: State,
    pub _gguf: Box<Gguf>,
    pub runtime: RuntimeCfg,
    pub metrics: EngineMetrics,
    pub last_was_text_prompt: bool,
}

pub struct LlmEngine {
    pub stop: AtomicBool,
    pub inner: Mutex<EngineInner>,
    // Read-only metadata accessible without locking `inner` during generation:
    pub arch: String,
    pub quant: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_ctx_reported: usize,
    pub has_tok_embd: bool,
}

impl LlmEngine {
    pub fn open(path: &str, n_ctx: usize, n_threads: usize) -> anyhow::Result<Box<LlmEngine>> {
        let g = Box::new(Gguf::open(Path::new(path))?);
        let fp = g.fingerprint();
        let file_mb = g.file_size() as f64 / (1024.0 * 1024.0);
        // SAFETY: `model` and `state` are declared before `_gguf` in `EngineInner`, so they are
        // dropped before `_gguf` unmaps the underlying GGUF memory map.
        let g_ref: &'static Gguf = unsafe { &*(g.as_ref() as *const Gguf) };
        let model = Model::load(g_ref, n_ctx, n_threads)?;
        let state = State::new(&model.cfg);

        let arch = model.cfg.arch.clone();
        let quant = model.primary_quant.to_string();
        let n_embd = model.cfg.n_embd;
        let n_layer = model.cfg.n_layer;
        let n_vocab = model.cfg.n_vocab;
        let n_ctx_reported = model.context_length_reported();
        let has_tok_embd = model.token_embd_tensor().is_ok();

        logcat::info(&format!(
            "loadModel: arch={arch} quant={quant} embd={n_embd} layers={n_layer} vocab={n_vocab} n_ctx={n_ctx_reported} simd={} fnv1a={fp:016x} size={file_mb:.1}MB",
            simd::simd_backend_name()
        ));

        Ok(Box::new(LlmEngine {
            stop: AtomicBool::new(false),
            inner: Mutex::new(EngineInner {
                model,
                state,
                _gguf: g,
                runtime: RuntimeCfg {
                    spec_decoding: true,
                    prefix_cache: true,
                    repeat_penalty: 1.10,
                },
                metrics: EngineMetrics {
                    last_prefill_tok_s: 0.0,
                    last_decode_tok_s: 0.0,
                    total_tokens_generated: 0,
                    spec_drafted: 0,
                    spec_accepted: 0,
                    prefix_cache_hits: 0,
                    fingerprint: fp,
                    file_size_mb: file_mb,
                },
                last_was_text_prompt: false,
            }),
            arch,
            quant,
            n_embd,
            n_layer,
            n_vocab,
            n_ctx_reported,
            has_tok_embd,
        }))
    }
}

// ============================================================================
// JNI Helper Utilities
// ============================================================================

#[inline(always)]
unsafe fn jni_fn(env: *mut JNIEnv) -> &'static JNINativeInterface {
    unsafe { &**env }
}

unsafe fn throw_java(env: *mut JNIEnv, cls_name: &str, msg: &str) {
    unsafe {
        let f = jni_fn(env);
        if (f.ExceptionCheck)(env) == JNI_TRUE {
            return;
        }
        if let (Ok(c_cls), Ok(c_msg)) = (CString::new(cls_name), CString::new(msg)) {
            let cls = (f.FindClass)(env, c_cls.as_ptr());
            if !cls.is_null() {
                (f.ThrowNew)(env, cls, c_msg.as_ptr());
                (f.DeleteLocalRef)(env, cls);
            }
        }
    }
}

unsafe fn jstring_to_rust(env: *mut JNIEnv, js: jstring) -> String {
    if js.is_null() {
        return String::new();
    }
    unsafe {
        let f = jni_fn(env);
        let len = (f.GetStringLength)(env, js);
        if len <= 0 {
            return String::new();
        }
        let chars = (f.GetStringChars)(env, js, std::ptr::null_mut());
        if chars.is_null() {
            return String::new();
        }
        let slice = std::slice::from_raw_parts(chars, len as usize);
        let s = String::from_utf16_lossy(slice);
        (f.ReleaseStringChars)(env, js, chars);
        s
    }
}

unsafe fn rust_to_jstring(env: *mut JNIEnv, s: &str) -> jstring {
    unsafe {
        let f = jni_fn(env);
        let utf16: Vec<u16> = s.encode_utf16().collect();
        (f.NewString)(env, utf16.as_ptr(), utf16.len() as jsize)
    }
}

unsafe fn engine_from_handle<'a>(env: *mut JNIEnv, handle: jlong) -> Option<&'a LlmEngine> {
    if handle == 0 {
        unsafe { throw_java(env, "java/lang/IllegalArgumentException", "invalid model handle") };
        return None;
    }
    Some(unsafe { &*(handle as usize as *const LlmEngine) })
}

struct CallbackRef {
    env: *mut JNIEnv,
    obj: jobject,
    cls: jclass,
    on_token: jmethodID,
    on_error: jmethodID,
    broken: bool,
}

impl CallbackRef {
    unsafe fn new(env: *mut JNIEnv, cb: jobject) -> Option<Self> {
        if cb.is_null() {
            return None;
        }
        unsafe {
            let f = jni_fn(env);
            let obj = (f.NewGlobalRef)(env, cb);
            let local_cls = (f.GetObjectClass)(env, cb);
            let cls = (f.NewGlobalRef)(env, local_cls);
            (f.DeleteLocalRef)(env, local_cls);
            let m_tok = CString::new("onToken").unwrap();
            let sig = CString::new("(Ljava/lang/String;)V").unwrap();
            let m_err = CString::new("onError").unwrap();
            let on_token = (f.GetMethodID)(env, cls, m_tok.as_ptr(), sig.as_ptr());
            let on_error = (f.GetMethodID)(env, cls, m_err.as_ptr(), sig.as_ptr());
            if on_token.is_null() {
                (f.DeleteGlobalRef)(env, obj);
                (f.DeleteGlobalRef)(env, cls);
                return None;
            }
            Some(CallbackRef {
                env,
                obj,
                cls,
                on_token,
                on_error,
                broken: false,
            })
        }
    }

    unsafe fn emit_token(&mut self, piece: &str) {
        if self.broken || piece.is_empty() {
            return;
        }
        unsafe {
            let f = jni_fn(self.env);
            let js = rust_to_jstring(self.env, piece);
            if js.is_null() {
                self.broken = true;
                return;
            }
            let arg = [jvalue { l: js }];
            (f.CallVoidMethodA)(self.env, self.obj, self.on_token, arg.as_ptr());
            (f.DeleteLocalRef)(self.env, js);
            if (f.ExceptionCheck)(self.env) == JNI_TRUE {
                self.broken = true;
                (f.ExceptionDescribe)(self.env);
                (f.ExceptionClear)(self.env);
            }
        }
    }

    unsafe fn emit_error(&mut self, msg: &str) {
        if self.broken || self.on_error.is_null() {
            return;
        }
        unsafe {
            let f = jni_fn(self.env);
            let js = rust_to_jstring(self.env, msg);
            if js.is_null() {
                self.broken = true;
                return;
            }
            let arg = [jvalue { l: js }];
            (f.CallVoidMethodA)(self.env, self.obj, self.on_error, arg.as_ptr());
            (f.DeleteLocalRef)(self.env, js);
            if (f.ExceptionCheck)(self.env) == JNI_TRUE {
                self.broken = true;
                (f.ExceptionDescribe)(self.env);
                (f.ExceptionClear)(self.env);
            }
        }
    }
}

impl Drop for CallbackRef {
    fn drop(&mut self) {
        unsafe {
            let f = jni_fn(self.env);
            if !self.obj.is_null() {
                (f.DeleteGlobalRef)(self.env, self.obj);
            }
            if !self.cls.is_null() {
                (f.DeleteGlobalRef)(self.env, self.cls);
            }
        }
    }
}

// ============================================================================
// Core Sampling + Speculative Decoding Loop (used by JNI & Rust tests)
// ============================================================================

pub fn run_decode_loop(
    engine: &LlmEngine,
    inner: &mut EngineInner,
    mut last_logits: Vec<f32>,
    n_predict: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    prompt_tokens: &[u32],
    mut emit: impl FnMut(&str) -> bool,
) -> anyhow::Result<usize> {
    if n_predict == 0 {
        return Ok(0);
    }
    let use_spec = inner.runtime.spec_decoding
        && temperature <= 0.0
        && !inner.model.has_delta
        && inner.model.cfg.is_causal;
    let mut sampler = Sampler {
        temp: temperature,
        top_k,
        top_p: if top_p > 0.0 && top_p <= 1.0 {
            top_p
        } else {
            1.0
        },
        repeat_penalty: inner.runtime.repeat_penalty,
        penalty_last_n: 64,
        rng: 0x9E3779B97F4A7C15 ^ (inner.state.pos as u64 + 1),
    };
    let mut generated: Vec<u32> = Vec::with_capacity(n_predict.min(1024));
    let mut drafter = NgramDrafter::new(
        K_SPEC_DRAFT_MAX,
        inner.model.tok.eos_ids(),
        inner.model.cfg.n_vocab,
    );
    if use_spec {
        drafter.begin(prompt_tokens);
    }
    let mut utf8 = Utf8StreamDecoder::default();
    let t_start = Instant::now();
    let mut produced = 0usize;

    while produced < n_predict && !engine.stop.load(Ordering::Relaxed) {
        let mut logits = last_logits;
        let tok = sampler.pick(&mut logits, &generated)? as u32;
        if produced == 0 {
            logcat::info(&format!("sampling: first token={tok}"));
        }
        if inner.model.tok.is_eos(tok) {
            break;
        }
        generated.push(tok);
        produced += 1;
        let bytes = inner.model.tok.token_bytes(tok, true);
        if let Some(piece) = utf8.push(&bytes)
            && !emit(&piece)
        {
            break;
        }
        if inner.state.pos + 1 >= inner.model.n_ctx || produced >= n_predict {
            break;
        }

        // Speculative draft when greedy + attention-only + enough context room
        let drafts = if use_spec {
            let room = (n_predict - produced).min(inner.model.n_ctx.saturating_sub(inner.state.pos + 2));
            let mut d = drafter.draft(tok);
            if d.len() > room {
                d.truncate(room);
            }
            d
        } else {
            Vec::new()
        };

        if drafts.is_empty() {
            let emb = inner.model.embed_token(tok);
            let (mut lg, _) = inner.model.step_embed_batch(&mut inner.state, vec![emb], false, true);
            if inner.last_was_text_prompt {
                inner.state.cached_tokens.push(tok);
            }
            if use_spec {
                drafter.accept(&[tok]);
            }
            last_logits = lg.pop().unwrap_or_default();
        } else {
            let k = drafts.len();
            inner.metrics.spec_drafted += k as u64;
            let base_pos = inner.state.pos;
            let mut batch_tokens = Vec::with_capacity(1 + k);
            batch_tokens.push(tok);
            batch_tokens.extend_from_slice(&drafts);
            let embs: Vec<Vec<f32>> = batch_tokens
                .iter()
                .map(|&id| inner.model.embed_token(id))
                .collect();
            let (all_logits, _) =
                inner
                    .model
                    .step_embed_batch(&mut inner.state, embs, true, true);

            let mut accepted = vec![tok];
            let mut next_logits = Vec::new();
            let mut hit_eos = false;
            let mut broken_cb = false;

            for i in 0..k {
                let mut lg_i = all_logits[i].clone();
                let picked = sampler.pick(&mut lg_i, &generated)? as u32;
                if picked == drafts[i] {
                    if inner.model.tok.is_eos(picked) {
                        hit_eos = true;
                        inner.state.truncate_attn(
                            base_pos + 1 + i,
                            inner.model.cfg.n_head_kv,
                            inner.model.cfg.head_dim,
                        );
                        break;
                    }
                    generated.push(picked);
                    accepted.push(picked);
                    produced += 1;
                    inner.metrics.spec_accepted += 1;
                    let bytes = inner.model.tok.token_bytes(picked, true);
                    if let Some(piece) = utf8.push(&bytes)
                        && !emit(&piece)
                    {
                        broken_cb = true;
                        break;
                    }
                    if produced >= n_predict {
                        inner.state.truncate_attn(
                            base_pos + 1 + i + 1,
                            inner.model.cfg.n_head_kv,
                            inner.model.cfg.head_dim,
                        );
                        break;
                    }
                } else {
                    // Mismatch at draft[i]: rollback KV to base_pos + 1 + i, then keep `picked` as
                    // the next token (evaluated in next iteration? No — `picked` hasn't been fed yet,
                    // wait: emit `picked`? Wait: if we emit `picked` right now, we must decode `picked`
                    // on the next step OR return `all_logits[i]` as `last_logits` so the outer loop
                    // picks `picked` cleanly without duplicating code!).
                    inner.state.truncate_attn(
                        base_pos + 1 + i,
                        inner.model.cfg.n_head_kv,
                        inner.model.cfg.head_dim,
                    );
                    next_logits = all_logits[i].clone();
                    break;
                }
            }

            if inner.last_was_text_prompt {
                inner.state.cached_tokens.extend_from_slice(&accepted);
            }
            if use_spec {
                drafter.accept(&accepted);
            }
            if hit_eos || broken_cb || produced >= n_predict {
                break;
            }
            if next_logits.is_empty() {
                // All k drafts matched -> final logits are at index k!
                next_logits = all_logits[k].clone();
            }
            last_logits = next_logits;
        }
    }

    if let Some(tail) = utf8.flush() {
        let _ = emit(&tail);
    }

    let dt = t_start.elapsed().as_secs_f64();
    if dt > 0.0 && produced > 0 {
        inner.metrics.last_decode_tok_s = produced as f64 / dt;
    }
    inner.metrics.total_tokens_generated += produced as u64;
    Ok(produced)
}

pub fn prefill_tokens(
    engine: &LlmEngine,
    inner: &mut EngineInner,
    tokens: &[u32],
) -> anyhow::Result<Vec<f32>> {
    if tokens.is_empty() {
        anyhow::bail!("empty prompt tokens");
    }
    let max_toks = inner.model.n_ctx.saturating_sub(4).max(1);
    let toks = if tokens.len() > max_toks {
        &tokens[..max_toks]
    } else {
        tokens
    };

    let can_prefix = inner.runtime.prefix_cache
        && !inner.model.has_delta
        && inner.model.cfg.is_causal
        && inner.last_was_text_prompt;

    let mut common = 0usize;
    if can_prefix {
        while common < inner.state.cached_tokens.len()
            && common < toks.len().saturating_sub(1)
            && inner.state.cached_tokens[common] == toks[common]
        {
            common += 1;
        }
    }

    let t_start = Instant::now();
    if common > 0 {
        inner.state.truncate_attn(
            common,
            inner.model.cfg.n_head_kv,
            inner.model.cfg.head_dim,
        );
        inner.metrics.prefix_cache_hits += common as u64;
    } else {
        inner.state.clear();
    }
    inner.last_was_text_prompt = true;

    let suffix = &toks[common..];
    let mut last_logits = Vec::new();
    let mut pos = 0usize;
    while pos < suffix.len() && !engine.stop.load(Ordering::Relaxed) {
        let chunk = K_PREFILL_CHUNK.min(suffix.len() - pos);
        let is_last = pos + chunk >= suffix.len();
        let embs: Vec<Vec<f32>> = suffix[pos..pos + chunk]
            .iter()
            .map(|&id| inner.model.embed_token(id))
            .collect();
        let (mut lg, _) = inner.model.step_embed_batch(&mut inner.state, embs, false, is_last);
        inner
            .state
            .cached_tokens
            .extend_from_slice(&suffix[pos..pos + chunk]);
        if is_last {
            last_logits = lg.pop().unwrap_or_default();
        }
        pos += chunk;
    }

    let dt = t_start.elapsed().as_secs_f64();
    if dt > 0.0 && !suffix.is_empty() {
        inner.metrics.last_prefill_tok_s = suffix.len() as f64 / dt;
    }
    Ok(last_logits)
}

pub fn prefill_embeddings(
    engine: &LlmEngine,
    inner: &mut EngineInner,
    flat: &[f32],
    count: usize,
    followup_tokens: &[u32],
) -> anyhow::Result<Vec<f32>> {
    inner.state.clear();
    inner.last_was_text_prompt = false;
    let d = inner.model.cfg.n_embd;
    let t_start = Instant::now();
    let followup_empty = followup_tokens.is_empty();
    let mut last_logits = Vec::new();
    let mut pos = 0usize;

    while pos < count && !engine.stop.load(Ordering::Relaxed) {
        let chunk = K_PREFILL_CHUNK.min(count - pos);
        let is_last = followup_empty && (pos + chunk >= count);
        let embs: Vec<Vec<f32>> = (0..chunk)
            .map(|i| {
                let s = (pos + i) * d;
                flat[s..s + d].to_vec()
            })
            .collect();
        let (mut lg, _) = inner.model.step_embed_batch(&mut inner.state, embs, false, is_last);
        if is_last {
            last_logits = lg.pop().unwrap_or_default();
        }
        pos += chunk;
    }

    if !followup_empty && !engine.stop.load(Ordering::Relaxed) {
        let max_follow = inner.model.n_ctx.saturating_sub(pos + 4);
        let follow = &followup_tokens[..followup_tokens.len().min(max_follow)];
        let mut p2 = 0usize;
        while p2 < follow.len() && !engine.stop.load(Ordering::Relaxed) {
            let chunk = K_PREFILL_CHUNK.min(follow.len() - p2);
            let is_last = p2 + chunk >= follow.len();
            let embs: Vec<Vec<f32>> = follow[p2..p2 + chunk]
                .iter()
                .map(|&id| inner.model.embed_token(id))
                .collect();
            let (mut lg, _) = inner.model.step_embed_batch(&mut inner.state, embs, false, is_last);
            if is_last {
                last_logits = lg.pop().unwrap_or_default();
            }
            p2 += chunk;
        }
    }

    let dt = t_start.elapsed().as_secs_f64();
    let total_toks = count + followup_tokens.len();
    if dt > 0.0 && total_toks > 0 {
        inner.metrics.last_prefill_tok_s = total_toks as f64 / dt;
    }
    Ok(last_logits)
}

// ============================================================================
// Exported JNI Entrypoints
// ============================================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn JNI_OnLoad(_vm: *mut JavaVM, _reserved: *mut c_void) -> jint {
    JNI_VERSION_1_6
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_loadModel(
    env: *mut JNIEnv,
    _thiz: jobject,
    path: jstring,
    n_ctx: jint,
    n_threads: jint,
) -> jlong {
    let file = unsafe { jstring_to_rust(env, path) };
    let n_ctx_req = if n_ctx > 0 { n_ctx as usize } else { 2048 };
    let threads = if n_threads > 0 { n_threads as usize } else { 2 };
    match LlmEngine::open(&file, n_ctx_req, threads) {
        Ok(engine) => Box::into_raw(engine) as usize as jlong,
        Err(e) => {
            logcat::err(&format!("loadModel failed: {e:#}"));
            unsafe {
                throw_java(env, "java/io/IOException", &format!("failed to load model: {e:#}"))
            };
            0
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_freeModel(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
) {
    if handle == 0 {
        unsafe { throw_java(env, "java/lang/IllegalArgumentException", "invalid model handle") };
        return;
    }
    let ptr = handle as usize as *mut LlmEngine;
    unsafe { drop(Box::from_raw(ptr)) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_getEmbeddingDim(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
) -> jint {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return -1;
    };
    if e.has_tok_embd {
        e.n_embd as jint
    } else {
        -1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_getContextLength(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
) -> jint {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return -1;
    };
    e.n_ctx_reported as jint
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_tokenize(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    text: jstring,
    add_special: jboolean,
    parse_special: jboolean,
) -> jintArray {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return std::ptr::null_mut();
    };
    let s = unsafe { jstring_to_rust(env, text) };
    let Ok(guard) = e.inner.lock() else {
        return std::ptr::null_mut();
    };
    let ids = guard
        .model
        .tok
        .encode(&s, add_special == JNI_TRUE, parse_special == JNI_TRUE);
    let jids: Vec<jint> = ids.into_iter().map(|id| id as jint).collect();
    unsafe {
        let f = jni_fn(env);
        let out = (f.NewIntArray)(env, jids.len() as jsize);
        if out.is_null() {
            return std::ptr::null_mut();
        }
        if !jids.is_empty() {
            (f.SetIntArrayRegion)(env, out, 0, jids.len() as jsize, jids.as_ptr());
        }
        out
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_tokenToPiece(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    token: jint,
) -> jstring {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return std::ptr::null_mut();
    };
    let Ok(guard) = e.inner.lock() else {
        return unsafe { rust_to_jstring(env, "") };
    };
    let piece = if token >= 0 {
        let bytes = guard.model.tok.token_bytes(token as u32, true);
        String::from_utf8_lossy(&bytes).into_owned()
    } else {
        String::new()
    };
    unsafe { rust_to_jstring(env, &piece) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_embedTokenRows(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    tokens: jintArray,
) -> jfloatArray {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return std::ptr::null_mut();
    };
    if !e.has_tok_embd {
        unsafe {
            throw_java(
                env,
                "java/lang/IllegalStateException",
                "model has no accessible token_embd",
            )
        };
        return std::ptr::null_mut();
    }
    unsafe {
        let f = jni_fn(env);
        let n = (f.GetArrayLength)(env, tokens);
        if n <= 0 {
            return (f.NewFloatArray)(env, 0);
        }
        let mut ids = vec![0 as jint; n as usize];
        (f.GetIntArrayRegion)(env, tokens, 0, n, ids.as_mut_ptr());
        let Ok(guard) = e.inner.lock() else {
            return std::ptr::null_mut();
        };
        let u_ids: Vec<u32> = ids.into_iter().map(|v| v.max(0) as u32).collect();
        let flat = guard.model.embed_tokens_flat(&u_ids);
        let out = (f.NewFloatArray)(env, flat.len() as jsize);
        if out.is_null() {
            return std::ptr::null_mut();
        }
        (f.SetFloatArrayRegion)(env, out, 0, flat.len() as jsize, flat.as_ptr());
        out
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_embedText(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    text: jstring,
    mode: jint,
) -> jfloatArray {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return std::ptr::null_mut();
    };
    let s = unsafe { jstring_to_rust(env, text) };
    let Ok(mut guard) = e.inner.lock() else {
        return std::ptr::null_mut();
    };
    let mut ids = guard.model.tok.encode(&s, true, true);
    if ids.is_empty() {
        return unsafe { (jni_fn(env).NewFloatArray)(env, 0) };
    }
    if ids.len() > 512 {
        ids.truncate(512);
    }

    let mean: Vec<f32> = if mode == 1 {
        guard.state.clear();
        guard.last_was_text_prompt = false;
        let embs: Vec<Vec<f32>> = ids
            .iter()
            .map(|&id| guard.model.embed_token(id))
            .collect();
        let inner = &mut *guard;
        let (_, hids) = inner
            .model
            .step_embed_batch(&mut inner.state, embs, false, false);
        let d = inner.model.cfg.n_embd;
        let mut acc = vec![0f64; d];
        let count = hids.len().max(1) as f64;
        for h in hids {
            for (a, v) in acc.iter_mut().zip(h) {
                *a += v as f64;
            }
        }
        acc.into_iter().map(|v| (v / count) as f32).collect()
    } else {
        let d = guard.model.cfg.n_embd;
        let flat = guard.model.embed_tokens_flat(&ids);
        let mut acc = vec![0f64; d];
        for row in flat.chunks_exact(d) {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += v as f64;
            }
        }
        let n = ids.len() as f64;
        acc.into_iter().map(|v| (v / n) as f32).collect()
    };

    unsafe {
        let f = jni_fn(env);
        let out = (f.NewFloatArray)(env, mean.len() as jsize);
        if out.is_null() {
            return std::ptr::null_mut();
        }
        (f.SetFloatArrayRegion)(env, out, 0, mean.len() as jsize, mean.as_ptr());
        out
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_generate(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    prompt: jstring,
    n_predict: jint,
    temperature: jfloat,
    top_k: jint,
    top_p: jfloat,
    callback: jobject,
) -> jint {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return -1;
    };
    let s = unsafe { jstring_to_rust(env, prompt) };
    let mut cb = unsafe { CallbackRef::new(env, callback) };
    let Ok(mut guard) = e.inner.lock() else {
        return -1;
    };
    e.stop.store(false, Ordering::SeqCst);

    let tokens = guard.model.tok.encode(&s, true, true);
    if tokens.is_empty() {
        return 0;
    }

    let last_logits = match prefill_tokens(e, &mut guard, &tokens) {
        Ok(l) => l,
        Err(err) => {
            if let Some(c) = cb.as_mut() {
                unsafe { c.emit_error(&format!("{err:#}")) };
            }
            return -1;
        }
    };

    if e.stop.load(Ordering::Relaxed) || last_logits.is_empty() {
        return 0;
    }

    let res = run_decode_loop(
        e,
        &mut guard,
        last_logits,
        n_predict.max(0) as usize,
        temperature,
        top_k.max(0) as usize,
        top_p,
        &tokens,
        |piece| {
            if let Some(c) = cb.as_mut() {
                unsafe { c.emit_token(piece) };
                !c.broken
            } else {
                true
            }
        },
    );

    match res {
        Ok(produced) => produced as jint,
        Err(err) => {
            logcat::err(&format!("generate error: {err:#}"));
            if let Some(c) = cb.as_mut() {
                unsafe { c.emit_error(&format!("{err:#}")) };
            }
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_generateFromEmbeddings(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    vectors: jfloatArray,
    count: jint,
    followup_prompt: jstring,
    n_predict: jint,
    temperature: jfloat,
    top_k: jint,
    top_p: jfloat,
    callback: jobject,
) -> jint {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return -1;
    };
    if count <= 0 {
        return 0;
    }
    let expected = (count as usize) * e.n_embd;
    let flat = unsafe {
        let f = jni_fn(env);
        let len = (f.GetArrayLength)(env, vectors) as usize;
        if len < expected {
            throw_java(
                env,
                "java/lang/IllegalArgumentException",
                "embedding array too small",
            );
            return -1;
        }
        let mut buf = vec![0f32; expected];
        (f.GetFloatArrayRegion)(env, vectors, 0, expected as jsize, buf.as_mut_ptr());
        buf
    };

    let followup = unsafe { jstring_to_rust(env, followup_prompt) };
    let mut cb = unsafe { CallbackRef::new(env, callback) };
    let Ok(mut guard) = e.inner.lock() else {
        return -1;
    };
    e.stop.store(false, Ordering::SeqCst);

    let followup_tokens = if followup.is_empty() {
        Vec::new()
    } else {
        guard.model.tok.encode(&followup, true, true)
    };

    let last_logits =
        match prefill_embeddings(e, &mut guard, &flat, count as usize, &followup_tokens) {
            Ok(l) => l,
            Err(err) => {
                if let Some(c) = cb.as_mut() {
                    unsafe { c.emit_error(&format!("{err:#}")) };
                }
                return -1;
            }
        };

    if e.stop.load(Ordering::Relaxed) || last_logits.is_empty() {
        return 0;
    }

    let res = run_decode_loop(
        e,
        &mut guard,
        last_logits,
        n_predict.max(0) as usize,
        temperature,
        top_k.max(0) as usize,
        top_p,
        &followup_tokens,
        |piece| {
            if let Some(c) = cb.as_mut() {
                unsafe { c.emit_token(piece) };
                !c.broken
            } else {
                true
            }
        },
    );

    match res {
        Ok(produced) => produced as jint,
        Err(err) => {
            logcat::err(&format!("generateFromEmbeddings error: {err:#}"));
            if let Some(c) = cb.as_mut() {
                unsafe { c.emit_error(&format!("{err:#}")) };
            }
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_stop(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
) {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return;
    };
    e.stop.store(true, Ordering::SeqCst);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_configureEngine(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
    spec_decoding: jboolean,
    prefix_cache: jboolean,
    repeat_penalty: jfloat,
) {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return;
    };
    if let Ok(mut guard) = e.inner.lock() {
        guard.runtime.spec_decoding = spec_decoding == JNI_TRUE;
        guard.runtime.prefix_cache = prefix_cache == JNI_TRUE;
        guard.runtime.repeat_penalty = if repeat_penalty >= 1.0 && repeat_penalty.is_finite() {
            repeat_penalty
        } else {
            1.0
        };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_clearKvCache(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
) {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return;
    };
    if let Ok(mut guard) = e.inner.lock() {
        guard.state.clear();
        guard.last_was_text_prompt = false;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_tree4five_gguf_LlmNative_getEngineInfoJson(
    env: *mut JNIEnv,
    _thiz: jobject,
    handle: jlong,
) -> jstring {
    let Some(e) = (unsafe { engine_from_handle(env, handle) }) else {
        return unsafe { rust_to_jstring(env, "{}") };
    };
    let json = if let Ok(guard) = e.inner.try_lock() {
        serde_json::json!({
            "engine": "RustingStrata",
            "version": "2.0.0",
            "arch": e.arch,
            "quant": e.quant,
            "n_embd": e.n_embd,
            "n_layer": e.n_layer,
            "n_vocab": e.n_vocab,
            "n_ctx": e.n_ctx_reported,
            "simd": simd::simd_backend_name(),
            "kv_used": guard.state.pos,
            "spec_decoding": guard.runtime.spec_decoding,
            "prefix_cache": guard.runtime.prefix_cache,
            "repeat_penalty": guard.runtime.repeat_penalty,
            "prefill_tok_s": (guard.metrics.last_prefill_tok_s * 100.0).round() / 100.0,
            "decode_tok_s": (guard.metrics.last_decode_tok_s * 100.0).round() / 100.0,
            "total_tokens": guard.metrics.total_tokens_generated,
            "spec_drafted": guard.metrics.spec_drafted,
            "spec_accepted": guard.metrics.spec_accepted,
            "prefix_cache_hits": guard.metrics.prefix_cache_hits,
            "fnv1a": format!("{:016x}", guard.metrics.fingerprint),
            "size_mb": (guard.metrics.file_size_mb * 10.0).round() / 10.0
        })
        .to_string()
    } else {
        // Generation currently holds lock — return static metadata + busy flag without blocking UI polling.
        serde_json::json!({
            "engine": "RustingStrata",
            "version": "2.0.0",
            "arch": e.arch,
            "quant": e.quant,
            "n_embd": e.n_embd,
            "n_layer": e.n_layer,
            "n_vocab": e.n_vocab,
            "n_ctx": e.n_ctx_reported,
            "simd": simd::simd_backend_name(),
            "busy": true
        })
        .to_string()
    };
    unsafe { rust_to_jstring(env, &json) }
}

// Keep unused libc/c_char silence check clean
#[allow(dead_code)]
fn _c_char_size() -> usize {
    std::mem::size_of::<c_char>()
}
