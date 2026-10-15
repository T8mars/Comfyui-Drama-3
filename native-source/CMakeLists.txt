cmake_minimum_required(VERSION 3.26)
project(drama_r2t2_capi LANGUAGES CXX)
set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp ad6c66839af3c5646fba8c6c2e2087a1e4e38948 source")
set(LLAMA_BUILD_DIR "" CACHE PATH "Matching existing shared-library build")
if(NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "Supply the pinned llama.cpp source via LLAMA_CPP_DIR")
endif()
add_library(drama_r2t2 SHARED native_ext.cpp)
target_compile_features(drama_r2t2 PRIVATE cxx_std_17)
target_compile_options(drama_r2t2 PRIVATE /utf-8 /EHsc)
target_include_directories(drama_r2t2 PRIVATE "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include" "${LLAMA_CPP_DIR}/tools/mtmd")
target_link_libraries(drama_r2t2 PRIVATE
    "${LLAMA_BUILD_DIR}/llama_cpp/src/Release/llama.lib"
    "${LLAMA_BUILD_DIR}/llama_cpp/tools/mtmd/Release/mtmd.lib"
    "${LLAMA_BUILD_DIR}/llama_cpp/ggml/src/Release/ggml-base.lib")
set_target_properties(drama_r2t2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")
