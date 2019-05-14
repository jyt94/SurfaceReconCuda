#pragma once

#pragma once
#include <iostream>
#include <cuda_runtime.h>

#include "ParticleDataCuda.h"
#include <cuda_runtime.h>

class ColorGrid {
public:
	cfloat3 xmin;
	cfloat3 xmax;
	float cellWidth;
	cint3 cellResolution;
	int numCells;
	cfloat3 palette[3];
	
	//sampled at cell centers
	vector<cfloat3>* rgb;
	vector<float>* density;
	cfloat3* device_rgb;
	float* device_density;

	void SetSize(cfloat3 min, cfloat3 max, float cellWidth_) {
		xmin = min;
		xmax = max;
		cellWidth = cellWidth_;

		cellResolution = ceil((xmax - xmin) / cellWidth);
		numCells = cellResolution.prod();
		xmax = xmin + cellResolution * cellWidth;

		rgb = new vector<cfloat3>;
		density = new vector<float>;
		rgb->resize(numCells);
		density->resize(numCells);
	}

	void AllocateDeviceBuffer() {
		cudaMalloc(&device_rgb, sizeof(cfloat3)*numCells);
		cudaMalloc(&device_density, sizeof(float)*numCells);

	}
	
	void CopyToHost() {
		cudaMemcpy(rgb->data(), device_rgb, numCells * sizeof(cfloat3), cudaMemcpyDeviceToHost);
		cudaMemcpy(density->data(), device_density, numCells * sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	void Release() {
		delete rgb;
		delete density;
		cudaFree(device_rgb);
		cudaFree(device_density);
	}
};