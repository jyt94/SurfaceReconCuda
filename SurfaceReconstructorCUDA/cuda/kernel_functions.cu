

#include <cuda_runtime.h>
#include "ZIndexGrid.cuh"
#include "SPHHelper.hpp"


__constant__ ZIndexGridCUDA device_zgrid;
__constant__ int device_indexMap[1024];
__constant__ SPHHelper device_sphhelper;

__inline__ __device__ int GetCellHash(cint3 pCoord, int resolution) {
	if (pCoord.x<0 || pCoord.x>=resolution ||
		pCoord.y<0 || pCoord.y>=resolution ||
		pCoord.z<0 || pCoord.z>=resolution)
		return INVALID_CELL;

	int mappedIndex = device_indexMap[pCoord.x] |
		device_indexMap[pCoord.y]<<1 |
		device_indexMap[pCoord.z]<<2;
	return mappedIndex;
}

__global__ void ComputeParticleHash(ZIndexGridCUDA zgrid) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= zgrid.numParticles)
		return;
	auto & pos = zgrid.particles[i].pos;
	auto coord = GetCoordinate(pos, zgrid.xmin, zgrid.cellWidth);
	auto hash = GetCellHash(coord, zgrid.resolution);
	zgrid.particleHashes[i] = hash;
	zgrid.particleIndices[i] = i;
}


__global__ void ReorderDataAndFindCellStart(ZIndexGridCUDA zgrid) {

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	extern __shared__ uint sharedHash[];
	uint hash;

	if (index < zgrid.numParticles)
	{
		hash = zgrid.particleHashes[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = zgrid.particleHashes[index - 1];
		}
	}

	__syncthreads();


	if (index < zgrid.numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			if (hash!=INVALID_CELL)
				zgrid.startIndices[hash] = index;

			if (index > 0)
				zgrid.endIndices[sharedHash[threadIdx.x]] = index;
		}
		if (index == zgrid.numParticles - 1)
		{
			if (hash != INVALID_CELL)
				zgrid.endIndices[hash] = index + 1;
		}


		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = zgrid.particleIndices[index];
		auto p = zgrid.particles[sortedIndex];
		zgrid.reorderBuffer[index] = p;
	}
}



__global__ void ComputeColorField(
	ZIndexGridCUDA zgrid,
	float spacing,
	float infectRadius,
	float normThres,
	int neighborThres,
	int* surfaceParticleMark
) {
	uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=zgrid.numParticles)
		return;
	
	auto& xi = zgrid.particles[i].pos;
	auto coord = GetCoordinate(xi, zgrid.xmin, zgrid.cellWidth);
	auto hash = zgrid.particleHashes[i];

	int numNeighbors=0;
	cfloat3 normal;
	float vol = spacing*spacing*spacing;

	for (int xx=-1; xx<=1; xx++)
		for (int yy=-1; yy<=1; yy++)
			for (int zz=-1; zz<=1; zz++) {
				auto coord1 = coord + cint3(xx, yy, zz);
				auto hash1 = GetCellHash(coord1, zgrid.resolution);
				if(hash1 == INVALID_CELL)
					continue;

				int startIndex = zgrid.startIndices[hash1];
				int endIndex = zgrid.endIndices[hash1];
				for (int j=startIndex; j<endIndex; j++) {
					if(j==i)
						continue;
					auto& xj = zgrid.particles[j].pos;
					auto xij = xi - xj;
					auto d = xij.Norm();
					if(d>=infectRadius)
						continue;
					auto nablaw = device_sphhelper.CubicGradient(xij);
					normal += nablaw;
					numNeighbors++;
				}
			}
	normal = normal*vol;
	auto nnorm = normal.Norm();
	//printf("%d %d %f\n", i, numNeighbors, nnorm);
	if(nnorm > normThres || numNeighbors<neighborThres)
		surfaceParticleMark[i] = 1;
	else
		surfaceParticleMark[i] = 0;
}