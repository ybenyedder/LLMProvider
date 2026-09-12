package com.tree4five.gguf

import android.app.Application
import android.util.Log

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        // llama.cpp is built from source into libggufllm.so (app/src/main/cpp);
        // preload it once here so service startup fails fast with a clear log
        // if the native library is missing.
        if (!LlmNative.ensureLoaded()) {
            Log.e("App", "libggufllm.so could not be loaded")
        }
    }
}
