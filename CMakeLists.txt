cmake_minimum_required(VERSION 3.18)
project(graph_parallel_study LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

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

add_executable(test_opeadd_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)nmp tests/correctness/test_openmp.cpp)
target_link_libraries(test_openmp PRIVATE gps_seq gps_omp)
add_test(NAME openmp COMMAND test_openmp)

find_package(MPI)
if(MPI_CXX_FOUND)
  add_library(gps_mpi
    src/mpi/dist_graph.cpp
    src/mpi/bfs.cpp
    src/mpi/pagerank.cpp
    src/mpi/connected_components.cpp
    src/mpi/sssp.cpp
    src/mpi/triangle_counting.cpp)
  target_link_libraries(gps_mpi PUBLIC gps_common MPI::MPI_CXX)

  add_executable(gps_bench_mpi src/mpi_bench_main.cpp)
  target_link_libraries(gps_bench_mpi PRIVATE gps_mpi gps_seq)

  add_executable(test_mpi tests/correctness/test_mpi.cpp)
  target_link_libraries(test_mpi PRIVATE gps_mpi gps_seq)
else()
  message(STATUS "MPI not found: skipping MPI targets")
endif()

include(CheckLanguage)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# Note: -DCMAKE_CUDA_ARCHITECTURES=89 for a 4090 or L4
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
endif()
set(CMAKE_CUDA_STANDARD 17)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(gps_cuda
    src/cuda/bfs.cu
    src/cuda/pagerank.cu
    src/cuda/connected_components.cu
    src/cuda/sssp.cu
    src/cuda/triangle_counting.cu)
  target_link_libraries(gps_cuda PUBLIC gps_common CUDA::cudart)

  add_executable(gps_bench_cuda src/cuda_bench_main.cpp)
  target_link_libraries(gps_bench_cuda PRIVATE gps_cuda gps_seq)

  add_executable(test_cuda tests/correctness/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE gps_cuda gps_seq)
else()
  message(STATUS "CUDA not found: skipping CUDA targets")
endif()