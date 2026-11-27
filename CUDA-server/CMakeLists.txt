cmake_minimum_required(VERSION 3.18)
project(cis5650_boids LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++20 for host and CUDA code.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

find_package(CCCL REQUIRED)

set(headers
    src/config.h
    src/glintVisualizer.h
    src/glintVisualizer.cuh
    src/packet_creation/glint.h
    src/packet_creation/compact.cuh
    src/packet_creation/culling.h
    src/cudaMat4.hpp
    src/glslUtility.hpp
    src/kernel.h
    src/main.hpp
    src/utilityCore.hpp
    )

set(sources
    src/packet_creation/glint.cu
    src/packet_creation/compact.cu
    src/packet_creation/culling.cu
    src/glslUtility.cpp
    src/camera.cpp
    src/globalCamera.cpp
    src/simulation.cu
    src/packet_creation/network.cu
    src/main.cpp
    src/utilityCore.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(${CMAKE_PROJECT_NAME} ${GL_LIBRARIES} CCCL::Thrust)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

add_custom_target(run_boids
    COMMAND ${CMAKE_COMMAND} -E env
        __GLX_VENDOR_LIBRARY_NAME=nvidia
        XDG_SESSION_TYPE=x11
        $<TARGET_FILE:cis5650_boids>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS cis5650_boids
)

# Select these targets in the IDE to build/run a separate Release configuration
# without changing the active Debug cache or its Build All behavior.
set(BOIDS_RELEASE_BUILD_DIR "${CMAKE_BINARY_DIR}/release")
set(BOIDS_RELEASE_CONFIGURE_ARGS
    -S "${CMAKE_SOURCE_DIR}"
    -B "${BOIDS_RELEASE_BUILD_DIR}"
    -G "${CMAKE_GENERATOR}"
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
)
if(CMAKE_CUDA_HOST_COMPILER)
    list(APPEND BOIDS_RELEASE_CONFIGURE_ARGS
        "-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")
endif()
add_custom_target(build_release
    COMMAND ${CMAKE_COMMAND} ${BOIDS_RELEASE_CONFIGURE_ARGS}
    COMMAND ${CMAKE_COMMAND} --build "${BOIDS_RELEASE_BUILD_DIR}"
        --config Release --target cis5650_boids --parallel
    USES_TERMINAL
    VERBATIM
)
add_custom_target(run_boids_release
    COMMAND ${CMAKE_COMMAND} --build "${BOIDS_RELEASE_BUILD_DIR}"
        --config Release --target run_boids
    DEPENDS build_release
    USES_TERMINAL
    VERBATIM
)

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
