cmake_minimum_required(VERSION 2.8)
project(Count-Pig)

# set CUDA
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch=compute_72,code=sm_72
	)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fpermissive -Wall")
set(CMAKE_CXX_STANDARD 11)
add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories("${PROJECT_SOURCE_DIR}/lib")

find_package(Boost COMPONENTS program_options filesystem REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "Could not locate Boost")
endif()

SET(CUDA_RT /usr/local/cuda-10.2/targets/${ARCH}-linux)
set(NVINFER_PATH /usr/lib/aarch64-linux-gnu/)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(sort SHARED src/kalman_filter.cpp  src/munkres.cpp src/track.cpp src/tracker.cpp src/utils.cpp) 
file(GLOB detectnetSources *.cpp common/logger.cpp lib/gpu_nms.cpp lib/nms_kernel.cu )
file(GLOB detectnetIncludes *.h lib/gpu_nms.hpp)

cuda_add_executable(detectnet ${detectnetSources})
target_link_libraries(detectnet ${OpenCV_LIBS} stdc++fs  ${Boost_LIBRARIES} sort)
target_link_libraries(detectnet ${NVINFER_PATH}/libnvinfer.so)
target_link_libraries(detectnet ${NVINFER_PATH}/libnvinfer_plugin.so)
target_link_libraries(detectnet ${NVINFER_PATH}/libnvparsers.so)
target_link_libraries(detectnet ${NVINFER_PATH}/libnvonnxparser.so)
