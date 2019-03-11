#pragma once
#include "ZIndexGrid.hpp"
#include "ParticleDataCuda.cuh"
#include <cuda_runtime.h>

class ZIndexGridCUDA{
public:

	float cellWidth;
	cfloat3 xmin;
	cfloat3 xmax;
	int resolution;
	int numParticles;
	int numCells;

	//device pointers
	uint* startIndices;
	uint* endIndices;
	uint* particleHashes;
	uint* particleIndices;
	Particle* particles;
	Particle* reorderBuffer;

	ZIndexGridCUDA(){}

	void SetSize(cfloat3 min, cfloat3 max, float width) {
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

	void AllocateDeviceBuffer();

	void BindParticles(ParticleDataCUDA& particleData) {
		particles = particleData.device_particles;
		numParticles = particleData.size();
		cudaMalloc(&particleHashes, sizeof(uint)*numParticles);
		cudaMalloc(&particleIndices, sizeof(uint)*numParticles);
		cudaMalloc(&reorderBuffer, sizeof(Particle)*numParticles);
	}

	void SortParticles();
};

