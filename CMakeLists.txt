cmake_minimum_required(VERSION 3.5)
project(objectify CXX C)

set(common
  input.cpp
  tracking.cpp)

set(darknet_dir ${CMAKE_SOURCE_DIR}/darknet)
set(darknet_lib libdarknet.so)

add_custom_target(darknet_target ALL
                  WORKING_DIRECTORY ${darknet_dir}
                  COMMAND make
                  COMMAND cp ${darknet_lib} ../)

set(common_libs -L${darknet_dir} -ldarknet)
set(ffmpeg_libs avcodec avdevice avformat avutil swscale)

set(sources ${common} avtest.cpp)
add_executable(avtest ${sources})
add_dependencies(avtest darknet_target)
target_include_directories(avtest PRIVATE darknet/include)
target_link_libraries(avtest ${common_libs} GL cuda cudart X11 ${ffmpeg_libs})

set(sources ${common} hook.cpp)
add_library(glhook SHARED ${sources})
add_dependencies(glhook darknet_target)
target_include_directories(glhook PRIVATE darknet/include)
target_link_libraries(glhook ${common_libs})
target_compile_definitions(glhook PUBLIC)
