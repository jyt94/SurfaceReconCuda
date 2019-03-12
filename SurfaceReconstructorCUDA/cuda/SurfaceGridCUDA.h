
#pragma once
#include "SurfaceGrids.h"
#include "ParticleDataCuda.h"
#include <cuda_runtime.h>

class SurfaceGridCUDA {
public:
	cfloat3 xmin;
	cfloat3 xmax;
	float cellWidth;
	cint3 cellResolution;
	cint3 vertexResolution;
	int numCells;
	int numVertices;
	int numSurfaceVertices;
	
	int* surfaceIndices;
	vector<SurfaceVertex>* surfaceVertices;

	int* device_surfaceIndices;
	SurfaceVertex* device_surfaceVertices;

	void SetSize(cfloat3 min, cfloat3 max, float cellWidth_);

	void AllocateDeviceBuffer() {
		cudaMalloc(&device_surfaceIndices, sizeof(int)*vertexResolution.prod());
		device_surfaceVertices = NULL;
	}
	void CopyToDevice();
	void CopyToHost();

	void InsertSurfaceVertex(cint3 coord);

	void Release() {
		delete[] surfaceIndices;
		delete surfaceVertices;
		cudaFree(device_surfaceIndices);
		cudaFree(device_surfaceVertices);
	}
};