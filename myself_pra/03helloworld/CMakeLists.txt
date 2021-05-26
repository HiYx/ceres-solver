cmake_minimum_required(VERSION 2.8)

find_package(Ceres REQUIRED)
Find_Package(Eigen3 REQUIRED)


set(CMAKE_CXX_FLAGS "-std=c++14")
include_directories(${CERES_INCLUDE_DIRS} 
	    "/usr/include/eigen3/")

add_executable(helloworld helloworld.cpp)
target_link_libraries(helloworld Ceres::ceres)
