cmake_minimum_required(VERSION 3.15)

project(seal-experiment VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(SEAL 3.6 REQUIRED)

if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

function(setup_property TARGETS)
  message("Setup")
  foreach(target IN LISTS TARGETS)
    message("Setting ${target}...")
    set_target_properties("${target}"
      PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    )
    target_compile_features("${target}" PUBLIC cxx_std_17)
    target_compile_options("${target}"
      PUBLIC
        -Wall
        $<$<CONFIG:Debug>:-O0 -g3>           # -D CMAKE_BUILD_TYPE=Debug
        $<$<CONFIG:Release>:-O3>             # -D CMAKE_BUILD_TYPE=Release
        $<$<CONFIG:MinSizeRel>:-Os>          # -D CMAKE_BUILD_TYPE=MinSizeRel
        $<$<CONFIG:RelWithDebInfo>:-O3 -g3>  # -D CMAKE_BUILD_TYPE=RelWithDebInfo
    )
    target_link_options("${target}" PUBLIC -Wall)
    target_include_directories("${target}"
      PRIVATE ${CMAKE_SOURCE_DIR}/src
      PRIVATE ${CMAKE_SOURCE_DIR}/include
    )
  endforeach()
endfunction()

file(GLOB common_sources src/utils/*.cpp)

add_executable(gen_keys
  src/gen_keys.cpp
  ${common_sources}
)
add_executable(example
  src/example.cpp
  ${common_sources}
)
add_executable(network_sample
  src/network_sample.cpp
  ${common_sources}
)
add_executable(main
  src/main.cpp
  ${common_sources}
)

target_link_libraries(gen_keys
  SEAL::seal
  stdc++fs
)
target_link_libraries(example
  SEAL::seal
)
target_link_libraries(network_sample
  SEAL::seal
  cnn
)
target_link_libraries(main
  SEAL::seal
  cnn
)

add_subdirectory(src/cnn)

set(TARGETS
  gen_keys
  example
  network_sample
  main
)

setup_property("${TARGETS}")
