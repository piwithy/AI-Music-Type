add_executable(predict src/main.cpp)

target_link_libraries(predict PUBLIC au-file-processing)
target_link_libraries(predict PRIVATE nlohmann_json::nlohmann_json)