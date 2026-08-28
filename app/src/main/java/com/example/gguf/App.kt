package com.example.gguf

import android.app.Application

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        // Must be set BEFORE any class from java-llama.cpp is loaded
        System.setProperty("de.kherud.llama.tmpdir", cacheDir.absolutePath)
        System.setProperty("java.io.tmpdir", cacheDir.absolutePath)
        
        // Android natively installs .so files from jniLibs, so we load them manually here
        // to bypass the broken java-llama.cpp classpath resource extractor.
        try {
            System.loadLibrary("c++_shared")
            System.loadLibrary("omp")
            System.loadLibrary("ggml")
            System.loadLibrary("llama")
            System.loadLibrary("jllama")
        } catch (e: Exception) {
            e.printStackTrace()
        } catch (e: Error) {
            e.printStackTrace()
        }
    }
}
