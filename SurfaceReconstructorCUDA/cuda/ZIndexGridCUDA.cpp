#include "ZIndexGridCUDA.h"
#include "cuda_common.h"
#include "helper_cuda.h"


extern __constant__ int device_indexMap[1024];
extern __constant__ cint3 device_neighborCells[27];

extern int indexMap[1024];

void ZIndexGridCUDA::SetSize(cfloat3 min, cfloat3 max, float width) {
	xmin = min;
	xmax = max;
	cellWidth = width;
	cfloat3 span = xmax - xmin;
	cint3 res = ceil(span/cellWidth);
	int maxres = res.maxElement();
	maxres = pow(2, ceil(log2f(maxres)));

	resolution = maxres;
	xmax = xmin + maxres * cellWidth;
	printf("Z Grid resolution: %d\n", resolution);
	numCells = resolution * resolution * resolution;
}

void ZIndexGridCUDA::AllocateDeviceBuffer() {
	cudaMalloc(&startIndices, sizeof(uint)*numCells);
	cudaMalloc(&endIndices, sizeof(uint)*numCells);
	cudaMemcpyToSymbol(device_indexMap, indexMap, sizeof(indexMap));

	cint3 neighborCells[27];
	int count=0;
	for(int zz=-1; zz<=1; zz++)
		for(int yy=-1; yy<=1; yy++)
			for (int xx=-1; xx<=1; xx++) {
				neighborCells[count++] = cint3(xx,yy,zz);
			}
	cudaMemcpyToSymbol(device_neighborCells, neighborCells, sizeof(neighborCells));
}

void ZIndexGridCUDA::BindParticles(ParticleDataCUDA& particleData) {
	particles = particleData.device_particles;
	numParticles = particleData.size();
	cudaMalloc(&particleHashes, sizeof(uint)*numParticles);
	cudaMalloc(&particleIndices, sizeof(uint)*numParticles);
	cudaMalloc(&reorderBuffer, sizeof(Particle)*numParticles);
}

void ZIndexGridCUDA::Release() {
	cudaFree(startIndices);
	cudaFree(endIndices);
	cudaFree(particleHashes);
	cudaFree(particleIndices);
	cudaFree(reorderBuffer);
}