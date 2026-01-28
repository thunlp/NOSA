cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(arkvale_cpp LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

add_subdirectory(${CMAKE_SOURCE_DIR}/../../3rdparty/pybind11 ${CMAKE_BINARY_DIR}/pybind11)
file(GLOB CPP_SOURCES "src/*.cu")
pybind11_add_module(arkvale_cpp MODULE ${PYTORCH_CPP_SOURCES} ${CPP_SOURCES})

target_include_directories(arkvale_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/pybind11/include)
target_include_directories(arkvale_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/flashinfer/include)
target_include_directories(arkvale_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/cutlass/include)
target_include_directories(arkvale_cpp PRIVATE ${CMAKE_SOURCE_DIR}/../../3rdparty/cutlass/tools/util/include)
target_compile_options(arkvale_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

set_target_properties(arkvale_cpp
PROPERTIES CXX_STANDARD                        17
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       17
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_SOURCE_DIR}/../../3rdparty/raft/cpp/template/cmake/thirdparty/fetch_rapids.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/_deps/rapids-cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()
include(${CMAKE_SOURCE_DIR}/../../3rdparty/raft/cpp/template/cmake/thirdparty/get_raft.cmake)
           
target_link_libraries(arkvale_cpp PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Python pybind11::module raft::raft)