#pragma once

#include "catpaw/vec_define.h"
#include "Grids.h"
#include "ParticleDataCuda.h"
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

	void SetSize(cfloat3 min, cfloat3 max, float width);
	void AllocateDeviceBuffer();

	void BindParticles(ParticleDataCUDA& particleData);

	void Release();
};

