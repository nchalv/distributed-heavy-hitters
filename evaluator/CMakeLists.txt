cmake_minimum_required(VERSION 3.20)
project(heavyheads LANGUAGES C CXX)

# ---- Build config ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HH_SUPPRESS_WARNINGS "Suppress compiler warnings for local builds" ON)
option(HH_ENABLE_STRICT_WARNINGS "Enable strict warning set for hh_core" OFF)

# ---- Deps ----
find_package(ZLIB REQUIRED)
include(FetchContent)

# BLAKE3 (via helper)
include(cmake/FetchBlake3.cmake)

# zstr (gzip iostreams)
FetchContent_Declare(
  zstr
  GIT_REPOSITORY https://github.com/mateidavid/zstr.git
  GIT_TAG v1.0.5
)
FetchContent_MakeAvailable(zstr)

# nlohmann::json (single-header)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ---- BLAKE3 portable-only object lib (no SIMD backends to avoid toolchain issues) ----
add_library(blake3_obj OBJECT
  ${blake3_SOURCE_DIR}/c/blake3.c
  ${blake3_SOURCE_DIR}/c/blake3_dispatch.c
  ${blake3_SOURCE_DIR}/c/blake3_portable.c
)
target_include_directories(blake3_obj PUBLIC ${blake3_SOURCE_DIR}/c)
target_compile_definitions(blake3_obj PUBLIC
  BLAKE3_NO_SSE2
  BLAKE3_NO_SSE41
  BLAKE3_NO_AVX2
  BLAKE3_NO_AVX512
  BLAKE3_NO_NEON
)
set_property(TARGET blake3_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

# ---- Core library ----
add_library(hh_core
  # core
  src/core/hash.cpp
  src/core/arena_map.cpp
  src/core/normalize.cpp
  # io
  src/io/reader.cpp
  # sketches
  src/sketches/ss.cpp
  src/sketches/heavylocker.cpp
  src/sketches/chk.cpp
  # oracle reference
  src/exact/exact_head.cpp
  # oracle
  src/oracle/oracle_all.cpp
  # hybrid
  src/hybrid/hybrid.cpp
  # coordinator
  src/coord/coordinator.cpp
  # sizing
  src/sizing/sizing.cpp
  # bench helpers
  src/bench/metrics.cpp
  

  # blake3 object files
  $<TARGET_OBJECTS:blake3_obj>
)

target_include_directories(hh_core PUBLIC
  include
  ${zstr_SOURCE_DIR}/src
  ${blake3_SOURCE_DIR}/c
)

target_link_libraries(hh_core PUBLIC
  ZLIB::ZLIB
  nlohmann_json::nlohmann_json
)

# Warning policy
if (HH_SUPPRESS_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-w)
  elseif (MSVC)
    add_compile_options(/W0)
  endif()
elseif (HH_ENABLE_STRICT_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hh_core PRIVATE -Wall -Wextra -Wpedantic)
  elseif (MSVC)
    target_compile_options(hh_core PRIVATE /W4)
  endif()
endif()

# ---- Apps ----
add_executable(hh_run apps/hh_run.cpp)
target_link_libraries(hh_run PRIVATE hh_core)

add_executable(hh_bench apps/hh_bench.cpp)
target_link_libraries(hh_bench PRIVATE hh_core)

# ---- Tests ----
enable_testing()
add_executable(hh_tests tests/hh_tests.cpp)
target_link_libraries(hh_tests PRIVATE hh_core)
add_test(NAME hh_tests COMMAND hh_tests)

# ---- Clean target ----
if(CMAKE_SCRIPT_MODE_FILE)
    # This allows running: cmake -P CMakeLists.txt
    message(STATUS "Cleaning project...")
    file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/build)
    file(REMOVE ${CMAKE_SOURCE_DIR}/hh_run ${CMAKE_SOURCE_DIR}/hh_bench ${CMAKE_SOURCE_DIR}/test_basic)
else()
    # Regular build mode
    add_custom_target(project-clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/hh_run ${CMAKE_SOURCE_DIR}/hh_bench ${CMAKE_SOURCE_DIR}/test_basic
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Cleaning project from source directory"
    )
endif()

# ---- Build info ----
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary dir: ${CMAKE_BINARY_DIR}")
