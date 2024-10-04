cmake_minimum_required(VERSION 3.0...3.26)

project(hnswib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")

    set(CMAKE_CXX_FLAGS "-Ofast -DNDEBUG")
    set(CMAKE_C_FLAGS  "-Ofast -DNDEBUG")

elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")

    set(CMAKE_CXX_FLAGS "-O2 -g -DNDEBUG")
    set(CMAKE_C_FLAGS  "-O2 -g -DNDEBUG")

    elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

    set(CMAKE_CXX_FLAGS "-O0 -g")
    set(CMAKE_C_FLAGS "-O0 -g")

    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-stack-protector -fno-var-tracking ")

else ()
    set(default_build_type "RelWithDebInfo")

    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
            "Default BUILD_TYPE is ${default_build_type}" FORCE)    
endif ()


add_library(hnsw_core
        hnsw.cpp
)
add_executable(main main.cpp)
target_link_libraries(main hnsw_core)
