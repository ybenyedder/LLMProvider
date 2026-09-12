# LlmNative methods are looked up by name from C++ (Java_com_tree4five_gguf_...);
# R8 must not rename them, and the callback interface is implemented from JNI.
-keepclasseswithmembernames class com.tree4five.gguf.LlmNative {
    native <methods>;
}
-keep class com.tree4five.gguf.LlmNative$* { *; }
