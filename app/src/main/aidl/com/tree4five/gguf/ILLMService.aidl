package com.tree4five.gguf;

import com.tree4five.gguf.ILLMCallback;

interface ILLMService {
    /**
     * Starts text generation and streams the output to the provided callback.
     */
    oneway void generateTextStream(String prompt, ILLMCallback callback);
    oneway void stopGeneration();
    
    /**
     * Retrieves the current version of the LLM Provider application.
     */
    String getVersion();
}
