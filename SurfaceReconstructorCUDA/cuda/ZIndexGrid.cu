#include "ZIndexGrid.cuh"
#include "cuda_common.cuh"
#include "helper_cuda.h"


extern __constant__ int device_indexMap[1024];
extern int indexMap[1024];

void ZIndexGridCUDA::AllocateDeviceBuffer() {
	cudaMalloc(&startIndices, sizeof(uint)*numCells);
	cudaMalloc(&endIndices, sizeof(uint)*numCells);
	cudaMemcpyToSymbol(device_indexMap, indexMap, sizeof(indexMap));
}

