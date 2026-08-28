package com.example.gguf

import android.app.Application

class App : Application() {
    override fun onCreate() {
        super.onCreate()
        // Must be set BEFORE any class from java-llama.cpp is loaded
        System.setProperty("de.kherud.llama.tmpdir", cacheDir.absolutePath)
        System.setProperty("java.io.tmpdir", cacheDir.absolutePath)
    }
}
