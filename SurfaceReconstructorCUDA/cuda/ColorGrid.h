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
	cint3 vertexResolution;
	int numCells;
	int numVertices;
	

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
		vertexResolution = cellResolution + cint3(1, 1, 1);
		numVertices = vertexResolution.prod();

		rgb = new vector<cfloat3>;
		density = new vector<float>;
		rgb->resize(vertexResolution.prod());
		density->resize(numVertices);
	}

	void AllocateDeviceBuffer() {
		cudaMalloc(&device_rgb, sizeof(cfloat3)*vertexResolution.prod());
		cudaMalloc(&device_density, sizeof(float)*vertexResolution.prod());

	}
	
	void CopyToHost() {
		cudaMemcpy(rgb->data(), device_rgb, vertexResolution.prod() * sizeof(cfloat3), cudaMemcpyDeviceToHost);
		cudaMemcpy(density->data(), device_density, vertexResolution.prod() * sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	void Release() {
		delete rgb;
		delete density;
		cudaFree(device_rgb);
		cudaFree(device_density);
	}
};