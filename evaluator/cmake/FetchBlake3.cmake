include(FetchContent)
FetchContent_Declare(
  blake3
  GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
  GIT_TAG        1.5.4
)
FetchContent_MakeAvailable(blake3)
