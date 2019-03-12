

#include "kernel_functions.h"
#include "cuda_common.h"
#include "helper_cuda.h"

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