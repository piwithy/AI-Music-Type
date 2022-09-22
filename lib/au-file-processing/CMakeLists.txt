add_library(au-file-processing STATIC src/AUFile.cpp)
target_include_directories(au-file-processing PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")