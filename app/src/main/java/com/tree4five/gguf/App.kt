package com.tree4five.gguf

import android.app.Application
import android.util.Log

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        // Pure-Rust inference engine (RustingStrata v2.0) compiled into libggufllm.so.
        if (!LlmNative.ensureLoaded()) {
            Log.e("App", "libggufllm.so could not be loaded")
        }
    }
}
