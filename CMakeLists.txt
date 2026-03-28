cmake_minimum_required(VERSION 3.16)
project(TernaryInferenceEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler flags ────────────────────────────────────────────────────────
# -march=native enables AVX2 + FMA + AVX-VNNI on Intel 12th/13th-gen CPUs.
# You can replace -march=native with -mavx2 -mfma -mavxvnni for portability.
set(OPT_FLAGS "-O3 -march=native -funroll-loops -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── Static library: libengine ─────────────────────────────────────────────
add_library(engine STATIC engine.cpp)
target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(engine PRIVATE -O3 -march=native -funroll-loops -ffast-math)
target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)

# ── Executable: inference_demo ────────────────────────────────────────────
add_executable(inference_demo inference_main.cpp)
target_link_libraries(inference_demo PRIVATE engine)

# ── Optional: enable address/undefined-behaviour sanitisers for debug build
# cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE=ON ..
option(SANITIZE "Enable ASan + UBSan" OFF)
if(SANITIZE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(  -fsanitize=address,undefined)
endif()
