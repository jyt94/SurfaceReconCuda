
#include <cuda_runtime.h>

#include "ZIndexGridCUDA.h"
#include "SurfaceGridCUDA.h"
#include "ColorGrid.h"

#include "cuda_common.h"
#include "helper_cuda.h"


__global__ void ComputeParticleHash(ZIndexGridCUDA zgrid);
__global__ void ReorderDataAndFindCellStart(ZIndexGridCUDA zgrid);
__global__ void ComputeColorField(
	ZIndexGridCUDA zgrid,
	float spacing,
	float infectRadius,
	float normThres,
	int neighborThres,
	int* surfaceParticleMark
);
__global__ void ComputeScalarValues(
	ZIndexGridCUDA zgrid,
	SurfaceGridCUDA sgrid,
	float particleSpacing,
	float infectRadius
);


void ReorderDataAndFindCellStart_Host(ZIndexGridCUDA& zgrid) {
	int numBlocks, numThreads;
	computeBlockSize(zgrid.numParticles, 256, numBlocks, numThreads);

	cudaMemset(zgrid.startIndices, CELL_EMPTY, zgrid.numCells*sizeof(uint));
	int sharedMemSize = sizeof(int)*(numThreads+1);

	ReorderDataAndFindCellStart <<< numBlocks, numThreads, sharedMemSize>>>(zgrid);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: reorder data");
	cudaMemcpy(zgrid.particles, zgrid.reorderBuffer, sizeof(Particle)*zgrid.numParticles, cudaMemcpyDeviceToDevice);
}



void ComputeParticleHash_Host(ZIndexGridCUDA& zgrid) {

	int numBlocks, numThreads;
	computeBlockSize(zgrid.numParticles, 256, numBlocks, numThreads);

	ComputeParticleHash <<<numBlocks, numThreads>>> (zgrid);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed:compute particle hash");
}

void ComputeColorField_Host(
	ZIndexGridCUDA& zgrid,
	float spacing,
	float infectRadius,
	float normThres,
	int neighborThres,
	int* surfaceParticleMark) {
	
	int numBlocks, numThreads;
	computeBlockSize(zgrid.numParticles, 256, numBlocks, numThreads);

	ComputeColorField<<<numBlocks, numThreads>>>(zgrid,
		spacing,
		infectRadius,
		normThres,
		neighborThres,
		surfaceParticleMark);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed:compute color field");
}

void ComputeScalarValues_Host(
	ZIndexGridCUDA& zgrid,
	SurfaceGridCUDA& sgrid,
	float particleSpacing,
	float infectRadius
) {
	int numBlocks, numThreads;
	computeBlockSize(sgrid.numSurfaceVertices, 256, numBlocks, numThreads);

	ComputeScalarValues<<<numBlocks, numThreads>>>(
		zgrid,
		sgrid,
		particleSpacing,
		infectRadius);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed:compute scalar values");
}





#include "SPHHelper.hpp"
#include "MarchingCube.h"

extern __constant__ int device_indexMap[1024];
extern __constant__ SPHHelper device_sphhelper;
extern __constant__ cint3 device_neighborCells[27];

__inline__ __device__ int GetCellHash(cint3 pCoord, int resolution) {
	if (pCoord.x < 0 || pCoord.x >= resolution ||
		pCoord.y < 0 || pCoord.y >= resolution ||
		pCoord.z < 0 || pCoord.z >= resolution)
		return INVALID_CELL;

	int mappedIndex = device_indexMap[pCoord.x] |
		device_indexMap[pCoord.y] << 1 |
		device_indexMap[pCoord.z] << 2;
	return mappedIndex;
}

__global__ void  ComputeColorValues(
	ZIndexGridCUDA zgrid,
	ColorGrid cgrid,
	float particleSpacing,
	float infectRadius
) {
	uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= cgrid.numCells)
		return;

	cfloat3 rgb(0, 0, 0);
	Vecf3 vf;
	for (int t = 0; t < 3; t++) vf.x[t] = 0;
	float density = 0;

	auto coord = GetCoordinate(i, cgrid.cellResolution);
	auto xi = GetPosition(coord, cgrid.xmin, cgrid.cellWidth);
	auto ofs = cgrid.cellWidth*0.5f;
	xi += cfloat3(ofs, ofs, ofs);

	auto zcoord = GetCoordinate(xi, zgrid.xmin, zgrid.cellWidth);
	auto zhash = GetCellHash(zcoord, zgrid.resolution);
	if (zhash == INVALID_CELL){
		printf("invalid cell error in color interpolation\n");
		return;
	}

	float pVol = particleSpacing * particleSpacing * particleSpacing;

	for (int i = 0; i < 27; i++) {
		auto coord1 = zcoord + device_neighborCells[i];
		auto hash1 = GetCellHash(coord1, zgrid.resolution);
		if (hash1 == INVALID_CELL)
			continue;

		int startIndex = zgrid.startIndices[hash1];
		if (startIndex == CELL_EMPTY)
			continue;
		int endIndex = zgrid.endIndices[hash1];

		// for each neighboring particle
		for (int j = startIndex; j < endIndex; j++) {

			auto& xj = zgrid.particles[j].pos;
			auto& p = zgrid.particles[j];
			auto xij = xi - xj;
			auto d = xij.Norm();
			if (d >= infectRadius)
				continue;
			float w_ij = device_sphhelper.Cubic(d) * pVol;
			for (int t = 0; t < 3; t++)
				vf.x[t] += p.vf.x[t] * w_ij;
			density += w_ij;
		}
	}

	//normalize
	auto sum = vf.x[0]+vf.x[1]+vf.x[2];
	if (sum < 1e-6) {
		vf.x[0] = vf.x[1] = vf.x[2] = 0;
		density = 0;
	}
	else {
		for (int i = 0; i < 3; i++) vf.x[i] /= sum;
		//if(i%100==0)
		//	printf("%f\n", density);
	}
	//compute color
	rgb += cgrid.palette[0] * vf.x[0];
	rgb += cgrid.palette[1] * vf.x[1];
	rgb += cgrid.palette[2] * vf.x[2];

	cgrid.device_rgb[i] = rgb;
	cgrid.device_density[i] = density;
}


void ComputeColorValues_Host(
	ZIndexGridCUDA& zgrid,
	ColorGrid& cgrid,
	float particleSpacing,
	float infectRadius
) {
	
	int numBlocks, numThreads;
	computeBlockSize(cgrid.numCells, 256, numBlocks, numThreads);

	ComputeColorValues << <numBlocks, numThreads >> > (
		zgrid,
		cgrid,
		particleSpacing,
		infectRadius);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed:compute color values");


}