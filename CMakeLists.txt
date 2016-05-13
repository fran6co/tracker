cmake_minimum_required(VERSION 3.5)
project(tracker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")

find_package(OpenCV 3.0 REQUIRED COMPONENTS core video imgproc imgcodecs highgui videoio calib3d)

set(SOURCE_FILES main.cpp Tracker.cpp Tracker.h HungarianMatching.cpp HungarianMatching.h)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(tracker ${SOURCE_FILES})
target_link_libraries(tracker
        ${OpenCV_LIBRARIES}
)

