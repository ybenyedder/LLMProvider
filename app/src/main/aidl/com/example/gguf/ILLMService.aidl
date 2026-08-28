package com.example.gguf;

import com.example.gguf.ILLMCallback;

interface ILLMService {
    /**
     * Starts text generation and streams the output to the provided callback.
     */
    oneway void generateTextStream(String prompt, ILLMCallback callback);
}
