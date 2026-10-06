# Rescorer library targets. Each backend is an option() block so they can be
# added independently (RESCORER_SLOTH, RESCORER_ELECTRA, RESCORER_QWEN).
#
# Usable standalone:  cmake -S Source/Engine/Rescorer -B build-rescorer-sloth
# or via add_subdirectory(Rescorer) from Source/Engine.

cmake_minimum_required(VERSION 3.16)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(McBopomofoRescorer CXX)
    set(RESCORER_STANDALONE ON)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    option(ENABLE_TEST "Build Test" ON)
    if (ENABLE_TEST)
        enable_testing()
        find_package(GTest QUIET)
        if (NOT GTest_FOUND)
            if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
                cmake_policy(SET CMP0135 NEW)
            endif ()
            include(FetchContent)
            FetchContent_Declare(googletest
                URL "https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip")
            FetchContent_MakeAvailable(googletest)
        endif ()
    endif ()
endif ()

# ---------------------------------------------------------------------------
# SlothLM-E (onnxruntime)
# ---------------------------------------------------------------------------
option(RESCORER_SLOTH "Build SlothRescorer (SlothLM-E via onnxruntime)" ON)
if (RESCORER_SLOTH)
    # onnxruntime v1.30.0 osx-arm64 prebuilt, unpacked under <repo>/vendor.
    set(ONNXRUNTIME_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../../../../vendor/onnxruntime/onnxruntime-osx-arm64-1.30.0"
        CACHE PATH "onnxruntime prebuilt package root (include/ and lib/)")
    set(SLOTH_MODEL_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../../../../bench/out/_models/sloth"
        CACHE PATH "SlothLM-E model directory used by sloth_rescorer_test")

    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

    add_library(sloth_rescorer STATIC sloth_rescorer.h sloth_rescorer.cpp rescorer.h)
    target_include_directories(sloth_rescorer
        PUBLIC "${CMAKE_CURRENT_LIST_DIR}"
        PRIVATE "${ONNXRUNTIME_ROOT}/include")
    target_link_libraries(sloth_rescorer PUBLIC "${ONNXRUNTIME_LIB}")
    target_compile_features(sloth_rescorer PUBLIC cxx_std_17)

    if (ENABLE_TEST)
        add_executable(sloth_rescorer_test sloth_rescorer_test.cpp)
        target_link_libraries(sloth_rescorer_test sloth_rescorer GTest::gtest_main)
        target_compile_definitions(sloth_rescorer_test PRIVATE
            SLOTH_MODEL_DIR="${SLOTH_MODEL_DIR}"
            SLOTH_FIXTURE="${CMAKE_CURRENT_LIST_DIR}/testdata/sloth_fixture.tsv")
        # Let the test binary find libonnxruntime.dylib without installing it.
        set_target_properties(sloth_rescorer_test PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_BUILD_RPATH}")
        include(GoogleTest)
        gtest_discover_tests(sloth_rescorer_test)
        add_custom_target(runSlothRescorerTest
            COMMAND $<TARGET_FILE:sloth_rescorer_test>
            DEPENDS sloth_rescorer_test)
    endif ()
endif ()

# ---------------------------------------------------------------------------
# ELECTRA-small discriminator (onnxruntime, int8) + ensemble combiner
# ---------------------------------------------------------------------------
option(RESCORER_ELECTRA "Build ElectraRescorer (hfl ELECTRA-small via onnxruntime)" ON)
set(ELECTRA_MODEL_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../models/electra"
    CACHE PATH "ELECTRA model directory (model_int8.onnx, vocab.txt) used by electra_rescorer_test")
if (RESCORER_ELECTRA)
    set(ONNXRUNTIME_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../../../../vendor/onnxruntime/onnxruntime-osx-arm64-1.30.0"
        CACHE PATH "onnxruntime prebuilt package root (include/ and lib/)")
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

    add_library(electra_rescorer STATIC electra_rescorer.h electra_rescorer.cpp rescorer.h)
    target_include_directories(electra_rescorer
        PUBLIC "${CMAKE_CURRENT_LIST_DIR}"
        PRIVATE "${ONNXRUNTIME_ROOT}/include")
    target_link_libraries(electra_rescorer PUBLIC "${ONNXRUNTIME_LIB}")
    target_compile_features(electra_rescorer PUBLIC cxx_std_17)
endif ()

# Ensemble combiner has no model dependency (z-score + weights + fallback).
option(RESCORER_ENSEMBLE "Build EnsembleRescorer (combines Rescorers)" ON)
if (RESCORER_ENSEMBLE)
    add_library(ensemble_rescorer STATIC ensemble_rescorer.h ensemble_rescorer.cpp rescorer.h)
    target_include_directories(ensemble_rescorer PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
    target_compile_features(ensemble_rescorer PUBLIC cxx_std_17)
    find_package(Threads REQUIRED)
    target_link_libraries(ensemble_rescorer PUBLIC Threads::Threads)
endif ()

if (ENABLE_TEST AND RESCORER_ENSEMBLE)
    include(GoogleTest)
    if (RESCORER_ELECTRA)
        add_executable(electra_rescorer_test electra_rescorer_test.cpp)
        target_link_libraries(electra_rescorer_test electra_rescorer GTest::gtest_main)
        target_compile_definitions(electra_rescorer_test PRIVATE
            ELECTRA_MODEL_DIR="${ELECTRA_MODEL_DIR}"
            ELECTRA_FIXTURE="${CMAKE_CURRENT_LIST_DIR}/testdata/electra_fixture.tsv")
        set_target_properties(electra_rescorer_test PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_BUILD_RPATH}")
        gtest_discover_tests(electra_rescorer_test)
    endif ()
    add_executable(ensemble_rescorer_test ensemble_rescorer_test.cpp)
    target_link_libraries(ensemble_rescorer_test ensemble_rescorer GTest::gtest_main)
    if (RESCORER_ELECTRA AND RESCORER_SLOTH)
        target_link_libraries(ensemble_rescorer_test sloth_rescorer electra_rescorer)
        target_compile_definitions(ensemble_rescorer_test PRIVATE
            ENSEMBLE_WITH_REAL_MODELS
            SLOTH_MODEL_DIR="${SLOTH_MODEL_DIR}"
            ELECTRA_MODEL_DIR="${ELECTRA_MODEL_DIR}"
            DE_FIXTURE="${CMAKE_CURRENT_LIST_DIR}/testdata/de_fixture.tsv")
        set_target_properties(ensemble_rescorer_test PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_BUILD_RPATH}")
    endif ()
    gtest_discover_tests(ensemble_rescorer_test)
endif ()

# ---------------------------------------------------------------------------
# Qwen3-0.6B-Base (llama.cpp + Metal, statically linked)
# ---------------------------------------------------------------------------
option(RESCORER_QWEN "Build QwenRescorer (Qwen3-0.6B via llama.cpp)" ON)
if (RESCORER_QWEN)
    set(LLAMA_CPP_ROOT
        "${CMAKE_CURRENT_LIST_DIR}/../../../../vendor/llama.cpp"
        CACHE PATH "llama.cpp source tree")
    set(QWEN_NBEST "${CMAKE_CURRENT_LIST_DIR}/../../../../bench/out/nbest.jsonl")
    set(QWEN_PY_RESULT
        "${CMAKE_CURRENT_LIST_DIR}/../../../../bench/out/result_B_qwen3-0.6b-base-bf16_lam0.jsonl")
    set(QWEN_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../models"
        CACHE PATH "Directory holding Qwen3-0.6B-Base GGUF files")

    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    add_subdirectory("${LLAMA_CPP_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

    add_library(qwen_rescorer STATIC qwen_rescorer.h qwen_rescorer.cpp rescorer.h)
    target_include_directories(qwen_rescorer PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
    target_link_libraries(qwen_rescorer PUBLIC llama)
    target_compile_features(qwen_rescorer PUBLIC cxx_std_17)

    add_executable(qwen_rescorer_test qwen_rescorer_test.cpp)
    target_include_directories(qwen_rescorer_test PRIVATE "${LLAMA_CPP_ROOT}/vendor")
    target_link_libraries(qwen_rescorer_test qwen_rescorer)
    # Run: qwen_rescorer_test <gguf> <nbest.jsonl> <python_result.jsonl>
    add_custom_target(runQwenRescorerTest
        COMMAND $<TARGET_FILE:qwen_rescorer_test> "${QWEN_MODEL_DIR}/Qwen3-0.6B-Base-Q8_0.gguf"
                "${QWEN_NBEST}" "${QWEN_PY_RESULT}"
        DEPENDS qwen_rescorer_test)
endif ()

# ---------------------------------------------------------------------------
# rescorer_bench: end-to-end C++ parity benchmark CLI (needs both backends)
#   cmake -S Source/Engine/Rescorer -B build-bench -DENABLE_TEST=OFF
#   cmake --build build-bench --target rescorer_bench
# ---------------------------------------------------------------------------
option(RESCORER_BENCH "Build rescorer_bench CLI (Sloth + Qwen + real McBopomofoLM)" ON)
if (RESCORER_BENCH AND (RESCORER_SLOTH OR RESCORER_QWEN))
    set(_engine_dir "${CMAKE_CURRENT_LIST_DIR}/..")
    if (NOT TARGET McBopomofoLMLib)
        # Standalone configure: pull in the engine libraries (no engine tests).
        set(_saved_enable_test "${ENABLE_TEST}")
        set(ENABLE_TEST OFF)
        add_subdirectory("${_engine_dir}" "${CMAKE_CURRENT_BINARY_DIR}/engine" EXCLUDE_FROM_ALL)
        set(ENABLE_TEST "${_saved_enable_test}")
    endif ()
    add_executable(rescorer_bench tools/rescorer_bench.cpp)
    target_include_directories(rescorer_bench PRIVATE
        "${_engine_dir}" "${_engine_dir}/gramambular2" "${CMAKE_CURRENT_LIST_DIR}"
        "${LLAMA_CPP_ROOT}/vendor")
    target_link_libraries(rescorer_bench PRIVATE McBopomofoLMLib gramambular2_lib)
    # AFM Input/ directory (parent of AFMBopomofo/), with trailing slash.
    get_filename_component(_bench_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." REALPATH)
    target_compile_definitions(rescorer_bench PRIVATE BENCH_ROOT="${_bench_root}/")
    if (RESCORER_SLOTH)
        target_link_libraries(rescorer_bench PRIVATE sloth_rescorer)
        target_compile_definitions(rescorer_bench PRIVATE RESCORER_BENCH_WITH_SLOTH)
        set_target_properties(rescorer_bench PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_BUILD_RPATH}")
    endif ()
    if (RESCORER_ELECTRA AND RESCORER_ENSEMBLE)
        target_link_libraries(rescorer_bench PRIVATE electra_rescorer ensemble_rescorer)
        target_compile_definitions(rescorer_bench PRIVATE RESCORER_BENCH_WITH_ELECTRA)
        set_target_properties(rescorer_bench PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_BUILD_RPATH}")
    endif ()
    if (RESCORER_QWEN)
        target_link_libraries(rescorer_bench PRIVATE qwen_rescorer)
        target_compile_definitions(rescorer_bench PRIVATE RESCORER_BENCH_WITH_QWEN)
    endif ()
endif ()
