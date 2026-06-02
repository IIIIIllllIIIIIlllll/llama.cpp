# gguf-extract-meta: extract metadata-only GGUF from local file or URL

set(TARGET gguf-extract-meta)

add_executable(${TARGET} gguf-extract-meta.cpp)
target_link_libraries(${TARGET} PRIVATE ggml llama-common cpp-httplib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
