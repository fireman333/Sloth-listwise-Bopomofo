# Personalization library (personal-finetune change, design D2/D7/D9/D10/D12).
#   personalize_weights: weights.bin format, SHA-256, ONNX initializer scan,
#                        merge + per-channel int8 quantizer (no engine deps;
#                        also linked by sloth_rescorer and compiled into the IME)
#   personalize_lib:     pool reader / segments / split, gate statistics,
#                        temperature refit, N-best features (trainer helper)
# Added from Source/Engine/CMakeLists.txt (tests run in the engine ctest) and,
# without tests, from Source/Engine/Rescorer/CMakeLists.txt.

cmake_minimum_required(VERSION 3.16)

set(_pz_dir "${CMAKE_CURRENT_LIST_DIR}")
set(_pz_engine_dir "${CMAKE_CURRENT_LIST_DIR}/..")
set(_pz_root "${CMAKE_CURRENT_LIST_DIR}/../../../..")  # AFM Input/

add_library(personalize_weights STATIC
        personal_weights.h personal_weights.cpp
        onnx_initializers.h onnx_initializers.cpp
        quantize.h quantize.cpp)
target_include_directories(personalize_weights PUBLIC "${_pz_engine_dir}")
target_compile_features(personalize_weights PUBLIC cxx_std_17)

add_library(personalize_lib STATIC
        pool_reader.h pool_reader.cpp
        gate.h gate.cpp
        temperature.h temperature.cpp
        nbest_features.h nbest_features.cpp)
target_include_directories(personalize_lib
        PUBLIC "${_pz_engine_dir}" "${_pz_engine_dir}/gramambular2"
        PRIVATE "${_pz_root}/vendor/llama.cpp/vendor")
target_link_libraries(personalize_lib PUBLIC personalize_weights McBopomofoLMLib gramambular2_lib)
target_compile_features(personalize_lib PUBLIC cxx_std_17)

if (ENABLE_TEST)
    add_executable(personalize_test
            personalize_test.cpp
            pool_reader_test.cpp
            gate_test.cpp
            temperature_test.cpp
            nbest_features_test.cpp)
    target_include_directories(personalize_test PRIVATE "${_pz_root}/vendor/llama.cpp/vendor")
    target_link_libraries(personalize_test personalize_lib GTest::gtest_main)
    target_compile_definitions(personalize_test PRIVATE
            PERSONALIZE_TESTDATA="${_pz_dir}/testdata"
            RWALK_MODEL_DIR="${_pz_root}/train/ranker/export/clean_v1"
            DATA_TXT="${_pz_engine_dir}/../Data/data.txt")
    include(GoogleTest)
    gtest_discover_tests(personalize_test DISCOVERY_MODE PRE_TEST)
endif ()
