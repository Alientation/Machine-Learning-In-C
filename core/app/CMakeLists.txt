cmake_minimum_required(VERSION 3.15)
project(app VERSION 0.0.1 LANGUAGES C CXX)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Dependencies
set(RAYLIB_VERSION 5.0)
include(FetchContent)
FetchContent_Declare(
    raylib
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    URL https://github.com/raysan5/raylib/archive/refs/tags/${RAYLIB_VERSION}.tar.gz
)
FetchContent_MakeAvailable(raylib)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions("_DEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -D_DEBUG -gdwarf-2")
    add_compile_options(-gdwarf-2)
    add_link_options(-gdwarf-2)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG -gdwarf-2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

find_package(util REQUIRED CONFIG)
find_package(model REQUIRED CONFIG)

# Our Project

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE 
    # source files for the library
    src/app.c
    src/visualizer.c
    src/visutil.c
    src/drawpanel.c
    src/dataset.c
)

# sets the search paths for the include files
include(GNUInstallDirs)
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME} PUBLIC util::util model::model raylib)

