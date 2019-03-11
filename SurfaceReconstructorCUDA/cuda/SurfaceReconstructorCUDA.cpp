#include "SurfaceReconstructorCUDA.cuh"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include "cuda_common.cuh"
#include "helper_cuda.h"
#include "kernel_functions.cuh"

extern __constant__ ZIndexGridCUDA device_zgrid;
extern __constant__ int device_indexMap[1024];
extern __constant__ SPHHelper device_sphhelper;

void SurfaceReconstructorCUDA::SetupGrids() {
	cout<<"preparing grids...\n";
	
	//setup zgrids
	zGrid.BindParticles(particleData);
	cfloat3 min, max;
	min = particleData.xmin - padding*2;
	max = particleData.xmax + padding*2;
	zGrid.SetSize(min, max, infectRadius);
	zGrid.AllocateDeviceBuffer();
	zGrid.BindParticles(particleData);
	cudaMemcpyToSymbol(device_zgrid, &zGrid, sizeof(ZIndexGridCUDA));

	//setup surface grids
	min = particleData.xmin - padding;
	max = particleData.xmax + padding;
	surfaceGrid.SetSize(min, max, particleSpacing*0.5);
	surfaceGrid.AllocateDeviceBuffer();

	cudaMemcpyToSymbol(device_sphhelper, &sphHelper, sizeof(SPHHelper));
}


void SurfaceReconstructorCUDA::ExtractSurfaceParticles() {
	cout<<"extracting surface particles\n";

	SortParticles();
	ComputeColorFieldAndMarkParticles();
}


void SurfaceReconstructorCUDA::SortParticles() {
	ComputeParticleHash_Host(zGrid);

	thrust::sort_by_key(
		thrust::device_ptr<uint>(zGrid.particleHashes),
		thrust::device_ptr<uint>(zGrid.particleHashes + zGrid.numParticles),
		thrust::device_ptr<uint>(zGrid.particleIndices)
	);

	ReorderDataAndFindCellStart_Host(zGrid);
	//particleData.CopyFromDevice();
	//printf("moo\n");
}



void SurfaceReconstructorCUDA::ComputeColorFieldAndMarkParticles() {
	ComputeColorField_Host(zGrid, 
		particleSpacing,
		infectRadius,
		normThres,
		neighborThres,
		surfaceParticleMark);
}

void SurfaceReconstructorCUDA::ExtractSurfaceVertices() {
	cout<<"extracing surface vertices\n";
}


void SurfaceReconstructorCUDA::ComputeScalarValues() {
	cout<<"computing scalar values\n";
}

void SurfaceReconstructorCUDA::Triangulate() {
	cout<<"triangulating\n";
}

void SurfaceReconstructorCUDA::OutputMesh() {

}