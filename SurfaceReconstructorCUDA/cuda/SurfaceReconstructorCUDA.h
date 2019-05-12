#pragma once
#include <iostream>
#include <cuda_runtime.h>

#include "SurfaceReconstructor.hpp"
#include "ParticleDataCuda.h"
#include "ZIndexGridCUDA.h"
#include "SurfaceGridCUDA.h"
#include "ColorGrid.h"

class SurfaceReconstructorCUDA {
public:
	SPHHelper sphHelper;
	float particleSpacing;
	float infectRadius;
	cfloat3 padding;
	float surfaceCellWidth;
	float normThres;
	int neighborThres;
	float isoValue;
	string inputFile;
	string outFile;

	ParticleDataCUDA particleData;
	ZIndexGridCUDA zGrid;
	SurfaceGridCUDA surfaceGrid;
	ColorGrid colorGrid;

	string colorFileName;

	int* surfaceParticleMark;
	int* device_surfaceParticleMark;

	void LoadConfig(const char* config);

	void ExtractSurface();
	void ExtractColor();

	void LoadParticle();
	void SetupZGrid();
	void SetupSurfaceGrid();
	void SetupColorGrids();

	void ExtractSurfaceParticles();
	void SortParticles();
	void ComputeColorFieldAndMarkParticles();

	void ExtractSurfaceVertices();


	void ComputeScalarValues();
	void ComputeColorValues();
	void OutputColorValues();

	void Triangulate();

	void Release();

};