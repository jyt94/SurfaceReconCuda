
#pragma once
#include <cuda_runtime.h>

#include "ZIndexGrid.cuh"

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

void ReorderDataAndFindCellStart_Host(ZIndexGridCUDA& zgrid);
void ComputeParticleHash_Host(ZIndexGridCUDA& zgrid);
void ComputeColorField_Host(
	ZIndexGridCUDA& zgrid,
	float spacing,
	float infectRadius,
	float normThres,
	int neighborThres,
	int* surfaceParticleMark);