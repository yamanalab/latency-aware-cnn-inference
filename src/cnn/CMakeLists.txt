set(module_name cnn)

option(BUILD_SHARED_LIBS "build as a shared library" ON)

ENABLE_LANGUAGE(C)

find_package(HDF5 REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

file(GLOB sources *.cpp layers/*.cpp)

if (BUILD_SHARED_LIBS)
  add_library(${module_name} SHARED ${sources})
else()
  add_library(${module_name} STATIC ${sources})
endif()

target_include_directories(${module_name}
  PUBLIC ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(${module_name}
  ${HDF5_LIBRARIES}
  hdf5
  hdf5_cpp
  SEAL::seal
  Eigen3::Eigen
)
