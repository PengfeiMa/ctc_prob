cmake_minimum_required(VERSION 3.3)
project(magicmind_plugin)

###############################################################################
# Build Environment
###############################################################################
set(BANG_TARGET_CPU_ARCH ${TARGET_CPU_ARCH})
message("-- TARGET_CPU_ARCH=${TARGET_CPU_ARCH}")
set(BANG_TARGET_MLU_ARCH ${TARGET_MLU_ARCH})
message("-- TARGET_MLU_ARCH=${TARGET_MLU_ARCH}")
set(TOOLCHAIN_ROOT ${TOOLCHAIN_ROOT})
message("-- TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT}")
set(NEUWARE_HOME ${NEUWARE_HOME})
message("-- NEUWARE_HOME=${NEUWARE_HOME}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/test")
message("-- EXECUTABLE_OUTPUT_PATH=${EXECUTABLE_OUTPUT_PATH}")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
message("-- LIBRARY_OUTPUT_PATH=${LIBRARY_OUTPUT_PATH}")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -pthread -pipe")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_FLAGS} -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -fPIC -std=c++11 -pthread -pipe")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_FLAGS} -O3")

set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${ABI})

###############################################################################
# Neuware Environment
###############################################################################
if (EXISTS ${NEUWARE_HOME})
  include_directories("${NEUWARE_HOME}/include")
  link_directories("${NEUWARE_HOME}/lib64")
  link_directories("${NEUWARE_HOME}/lib")
  if (${TARGET_CPU_ARCH} MATCHES ".*aarch64.*")
    link_directories("${NEUWARE_HOME}/edge/lib64")
  endif()
else()
  message(FATAL_ERROR "NEUWARE cannot be found. Refer to README.md to prepare NEUWARE_HOME environment.")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../common/")

###############################################################################
# Build MagicMind Plugin
###############################################################################
file(GLOB_RECURSE common_src_files FOLLOW_SYMLINKS "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.cpp")
file(GLOB_RECURSE host_src_files FOLLOW_SYMLINKS "${CMAKE_CURRENT_SOURCE_DIR}/pluginops/*.cc")
message("-- CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message("-- host_src_file=${host_src_files}")

add_library(mm_plugin SHARED "${common_src_files}" "${host_src_files}")
target_link_libraries(mm_plugin cnrt cnnl_extra magicmind_runtime )

