# Native bridge for the Tree4Five GGUF LLM service.
#
# Builds a single shared library `ggufllm` linking our JNI code against the
# vendored llama.cpp (see llama.cpp/README.md for the vendored version).
cmake_minimum_required(VERSION 3.22)
project(ggufllm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force Release: ggml asserts abort the process in debug and -O0 is several
# times slower on the target tablet.
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)

# llama.cpp: core library only (examples/tests/tools are not even vendored,
# but keep the options OFF so the root CMakeLists never references those
# subdirectories). LLAMA_CURL was removed upstream (deprecated no-op).
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)

# ggml: Android/NDK friendly settings.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)    # no OpenMP runtime in the NDK; pthread pool
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)    # -march=native is invalid when cross-compiling
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

add_library(ggufllm SHARED gguf_llm_jni.cpp)

target_link_libraries(ggufllm PRIVATE llama log)

# --exclude-libs,ALL: do not re-export the static llama/ggml symbols from our .so
# (avoids clashing with any other libllama an app process might load).
# 16 KiB pages are required by newer Android targets.
target_link_options(ggufllm PRIVATE -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
