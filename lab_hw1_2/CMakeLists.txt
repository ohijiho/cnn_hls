cmake_minimum_required(VERSION 3.15)
project(lab_hw1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(main cnn.cpp config.cpp main.cpp)
