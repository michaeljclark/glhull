cmake_minimum_required (VERSION 3.12)

project(glhull VERSION 0.0.1)

include(CheckLibraryExists)
check_library_exists(m sqrtf "" HAVE_LIB_M)
if (HAVE_LIB_M)
    set(EXTRA_LIBS ${EXTRA_LIBS} m)
endif (HAVE_LIB_M)

find_package(PkgConfig)
pkg_check_modules(GLFW3 REQUIRED glfw3)

# Find OpenGL library
include(FindOpenGL)

# user configurable options
option(EXTERNAL_GLFW "Use external GLFW project" ON)
option(EXTERNAL_GLAD "Use external GLAD project" ON)

message(STATUS "Build external GLFW = ${EXTERNAL_GLFW}")
message(STATUS "Build external GLAD = ${EXTERNAL_GLAD}")

# Support for external glfw
if (EXTERNAL_GLFW)
    include(ExternalProject)
    ExternalProject_Add(
        GLFW
        GIT_REPOSITORY "https://github.com/glfw/glfw.git"
        GIT_TAG "3.3-stable"
        SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/glfw3"
        STEP_TARGETS build
        EXCLUDE_FROM_ALL TRUE
        CMAKE_ARGS -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF
                   -DGLFW_BUILD_DOCS=OFF -DGLFW_INSTALL=OFF
    )
    set(GLFW_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/third_party/glfw3/include")
    set(GLFW_LIBRARY_DIR "${CMAKE_BINARY_DIR}/GLFW-prefix/src/GLFW-build/src/")
    set(GLFW_LIBRARIES glfw3 pthread dl)
    include_directories(${GLFW_INCLUDE_DIRS})
    link_directories(${GLFW_LIBRARY_DIR})
    set(GLFW_LIBS_ALL ${GLFW_LIBRARIES} ${EXTRA_LIBS} X11)
else ()
    add_definitions(${GLFW3_CFLAGS})
    set(GLFW_LIBS_ALL ${GLFW3_LDFLAGS} ${EXTRA_LIBS})
endif ()

# Support for external glad
if (EXTERNAL_GLAD)
    include(ExternalProject)
    ExternalProject_Add(
        GLAD
        GIT_REPOSITORY "https://github.com/Dav1dde/glad.git"
        SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/glad"
        STEP_TARGETS build
        EXCLUDE_FROM_ALL TRUE
        CMAKE_ARGS -DGLAD_EXPORT=OFF
    )
    set(GLAD_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/GLAD-prefix/src/GLAD-build/include/")
    set(GLAD_LIBRARY_DIR "${CMAKE_BINARY_DIR}/GLAD-prefix/src/GLAD-build/")
    set(GLAD_LIBRARIES glad)
    include_directories(${GLAD_INCLUDE_DIRS})
    link_directories(${GLAD_LIBRARY_DIR})
    list(APPEND GLFW_LIBS_ALL ${GLAD_LIBRARIES})
else ()
    list(APPEND GLFW_LIBS_ALL ${OPENGL_opengl_LIBRARY})
endif ()

# fpng
include_directories(third_party/fpng)
add_library(fpng third_party/fpng/fpng.cpp third_party/fpng/fpng_c.cpp)
target_compile_definitions(fpng PRIVATE -DFPNG_NO_SSE=0 -D_LARGEFILE64_SOURCE=1 -D_FILE_OFFSET_BITS=64)
target_compile_options(fpng PRIVATE -fno-strict-aliasing -msse4.1 -mpclmul)
list(APPEND GLFW_LIBS_ALL fpng)

# Find FreeType2 library
find_package(PkgConfig)
pkg_check_modules(FT2 freetype2)
if (${FT2_FOUND})
  list(APPEND GLFW_LIBS_ALL ${FT2_LIBRARIES})
else()
  message(ERROR "can't find freetype2")
endif()

# macro to add opengl example
macro(gl_cprogram prog)
    target_link_libraries(${prog} ${GLFW_LIBS_ALL})
    if (EXTERNAL_GLAD)
        target_compile_definitions(${prog} PRIVATE -DHAVE_GLAD)
    endif ()
    if (EXTERNAL_GLFW)
        add_dependencies(${prog} GLFW-build)
    endif ()
    if (EXTERNAL_GLAD)
        add_dependencies(${prog} GLAD-build)
    endif ()
endmacro()

#
# glhull
#

include_directories(include)
include_directories(third_party/stb)
include_directories(third_party/nanovg)
include_directories(${FT2_INCLUDE_DIRS})

set(NVG third_party/nanovg)
set(NVG_SOURCE ${NVG}/nanovg.c ${NVG}/nanovg_perf.c)

add_executable(mbhull src/mbhull.c)
gl_cprogram(mbhull)

add_executable(glhull ${NVG_SOURCE} src/glhull.c)
gl_cprogram(glhull)

add_executable(gldemo ${NVG_SOURCE} src/gldemo.c)
gl_cprogram(gldemo)

add_executable(nanovg_demo ${NVG_SOURCE} ${NVG}/nanovg_demo.c)
gl_cprogram(nanovg_demo)
