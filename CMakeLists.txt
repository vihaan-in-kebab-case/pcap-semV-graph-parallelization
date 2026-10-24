cmake_minimum_required(VERSION 3.18)
project(graph_parallel_study LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(gps_common src/common/graph_loader.cpp)
target_include_directories(gps_common PUBLIC include)

add_library(gps_seq
  src/sequential/bfs.cpp
  src/sequential/pagerank.cpp
  src/sequential/connected_components.cpp
  src/sequential/sssp.cpp
  src/sequential/triangle_counting.cpp)
target_link_libraries(gps_seq PUBLIC gps_common)

find_package(OpenMP REQUIRED)
add_library(gps_omp
  src/openmp/bfs.cpp
  src/openmp/pagerank.cpp
  src/openmp/connected_components.cpp
  src/openmp/sssp.cpp
  src/openmp/triangle_counting.cpp)
target_link_libraries(gps_omp PUBLIC gps_common OpenMP::OpenMP_CXX)

add_executable(gps_bench src/bench_main.cpp)
target_link_libraries(gps_bench PRIVATE gps_seq gps_omp)

enable_testing()
add_executable(test_sequential tests/correctness/test_sequential.cpp)
target_link_libraries(test_sequential PRIVATE gps_seq)
add_test(NAME sequential COMMAND test_sequential)

add_executable(test_openmp tests/correctness/test_openmp.cpp)
target_link_libraries(test_openmp PRIVATE gps_seq gps_omp)
add_test(NAME openmp COMMAND test_openmp)