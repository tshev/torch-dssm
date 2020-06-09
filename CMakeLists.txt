cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dssm)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(dssm src/main.cpp)
target_link_libraries(dssm "${TORCH_LIBRARIES}")
set_property(TARGET dssm PROPERTY CXX_STANDARD 17)
