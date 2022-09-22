cmake_minimum_required(VERSION 3.13)
project(Embedded-ML LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall")

set(JSON_BuildTests OFF CACHE INTERNAL "")

add_subdirectory(lib/json-3.9.1)
add_subdirectory(lib/au-file-processing)
add_subdirectory(features_extraction)
add_subdirectory(predicting)