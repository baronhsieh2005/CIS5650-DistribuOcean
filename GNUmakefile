CMAKE_ALT1 := /usr/local/bin/cmake
CMAKE_ALT2 := /Applications/CMake.app/Contents/bin/cmake
CMAKE := $(shell \
	which cmake 2>/dev/null || \
	([ -e ${CMAKE_ALT1} ] && echo "${CMAKE_ALT1}") || \
	([ -e ${CMAKE_ALT2} ] && echo "${CMAKE_ALT2}") \
	)

CMAKE_C_COMPILER := /usr/bin/gcc-15
CMAKE_CXX_COMPILER := /usr/bin/g++-15
CMAKE_CUDA_HOST_COMPILER := /usr/bin/g++-15

CMAKE_FLAGS := \
	-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
	-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
	-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}

all: Release

Debug: build
	(cd build && ${CMAKE} ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=$@ .. && make)

MinSizeRel: build
	(cd build && ${CMAKE} ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=$@ .. && make)

Release: build
	(cd build && ${CMAKE} ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=$@ .. && make)

RelWithDebugInfo: build
	(cd build && ${CMAKE} ${CMAKE_FLAGS} -DCMAKE_BUILD_TYPE=$@ .. && make)

build:
	mkdir -p build

clean:
	((cd build && make clean) 2>&- || true)

.PHONY: all Debug MinSizeRel Release RelWithDebugInfo clean