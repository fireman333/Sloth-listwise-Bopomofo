# AFMPersonalTrainer — on-device R fine-tune helper (personal-finetune design
# D1/D5/D8): C++ + static CPU-only MLX 0.32.x + the engine (McBopomofoLM,
# gramambular2, SlothRescorer / R-walk, Personalize library).
#
#   ../bench/.venv/bin/cmake -S Source/PersonalTrainer -B build-trainer -G Ninja \
#     -DCMAKE_MAKE_PROGRAM=../bench/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
#     -DGTest_DIR=/opt/anaconda3/lib/cmake/GTest -DCMAKE_BUILD_RPATH=/opt/anaconda3/lib
#   ../bench/.venv/bin/cmake --build build-trainer && (cd build-trainer && ../../bench/.venv/bin/ctest)
#
# MLX: static CPU-only build (MLX_BUILD_METAL=OFF, BUILD_SHARED_LIBS=OFF) installed
# under <AFM Input>/vendor/mlx-0.32.3-cpu-static (see research 16 / tasks 3.1).

cmake_minimum_required(VERSION 3.16)
project(AFMPersonalTrainer CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
endif ()

option(ENABLE_TEST "Build trainer tests" ON)

get_filename_component(_tr_root "${CMAKE_CURRENT_LIST_DIR}/../../.." REALPATH)  # AFM Input/
set(_tr_engine "${CMAKE_CURRENT_LIST_DIR}/../Engine")
set(MLX_ROOT "${_tr_root}/vendor/mlx-0.32.3-cpu-static" CACHE PATH "static CPU-only MLX install prefix")
find_library(MLX_STATIC_LIB NAMES libmlx.a PATHS "${MLX_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ACCELERATE Accelerate REQUIRED)

if (ENABLE_TEST)
    enable_testing()
    find_package(GTest REQUIRED)
endif ()

# Engine libraries without their tests, then the Rescorer (SlothRescorer + precise walk).
set(_saved_enable_test "${ENABLE_TEST}")
set(ENABLE_TEST OFF)
add_subdirectory("${_tr_engine}" "${CMAKE_CURRENT_BINARY_DIR}/engine" EXCLUDE_FROM_ALL)
set(RESCORER_ELECTRA OFF CACHE BOOL "" FORCE)
set(RESCORER_ENSEMBLE OFF CACHE BOOL "" FORCE)
set(RESCORER_QWEN OFF CACHE BOOL "" FORCE)
set(RESCORER_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory("${_tr_engine}/Rescorer" "${CMAKE_CURRENT_BINARY_DIR}/rescorer" EXCLUDE_FROM_ALL)
set(ENABLE_TEST "${_saved_enable_test}")

set(_tr_json "${_tr_root}/vendor/llama.cpp/vendor")

add_library(trainer_core STATIC
        train_base.h train_base.cpp
        train_data.h train_data.cpp
        mlx_trainer.h mlx_trainer.cpp
        pipeline.h pipeline.cpp
        system_probes.h system_probes.mm)
target_include_directories(trainer_core
        PUBLIC "${CMAKE_CURRENT_LIST_DIR}" "${_tr_engine}" "${_tr_engine}/Rescorer" "${_tr_json}"
        PRIVATE "${MLX_ROOT}/include")
target_link_libraries(trainer_core PUBLIC
        personalize_lib precise_walk sloth_rescorer McBopomofoLMLib gramambular2_lib
        "${MLX_STATIC_LIB}" "${ACCELERATE}"
        "-framework Foundation" "-framework IOKit" "-framework CoreFoundation")
target_compile_options(trainer_core PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)

add_executable(AFMPersonalTrainer main.cpp)
target_link_libraries(AFMPersonalTrainer PRIVATE trainer_core)
# Inside the app: Contents/MacOS/AFMPersonalTrainer, ORT in Contents/Frameworks.
set_target_properties(AFMPersonalTrainer PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        INSTALL_RPATH "@executable_path/../Frameworks")
target_link_options(AFMPersonalTrainer PRIVATE "LINKER:-dead_strip"
        "LINKER:-rpath,@executable_path/../Frameworks")

if (ENABLE_TEST)
    add_executable(trainer_test trainer_test.cpp)
    target_include_directories(trainer_test PRIVATE "${MLX_ROOT}/include")
    target_link_libraries(trainer_test PRIVATE trainer_core GTest::gtest_main)
    target_compile_definitions(trainer_test PRIVATE
            TRAINER_TESTDATA="${CMAKE_CURRENT_LIST_DIR}/testdata"
            FIXTURE_BASE_FP32="${_tr_root}/research/spike_runtime/data/fixture_base_fp32.bin"
            RWALK_MODEL_DIR="${_tr_root}/train/ranker/export/clean_v1"
            PACK_DIR="${_tr_root}/train/personalize/pack/out/personalize"
            SIM_DATA_DIR="${_tr_root}/train/personalize/data"
            DATA_TXT="${_tr_engine}/../Data/data.txt"
            TRAINER_EXE="$<TARGET_FILE:AFMPersonalTrainer>")
    set_target_properties(trainer_test PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CMAKE_BUILD_RPATH}")
    add_dependencies(trainer_test AFMPersonalTrainer)
    # Task 3.2 parity run (not a ctest; see tools/trainer_parity.cpp).
    add_executable(trainer_parity tools/trainer_parity.cpp)
    target_include_directories(trainer_parity PRIVATE "${MLX_ROOT}/include")
    target_link_libraries(trainer_parity PRIVATE trainer_core)
    set_target_properties(trainer_parity PROPERTIES BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")
    include(GoogleTest)
    gtest_discover_tests(trainer_test DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60
            PROPERTIES TIMEOUT 900)
endif ()
