
#pragma once

#include "ZIndexGridCUDA.h"
#include "SurfaceGridCUDA.h"
#include "ColorGrid.h"


void ReorderDataAndFindCellStart_Host(ZIndexGridCUDA& zgrid);
void ComputeParticleHash_Host(ZIndexGridCUDA& zgrid);
void ComputeColorField_Host(
	ZIndexGridCUDA& zgrid,
	float spacing,
	float infectRadius,
	float normThres,
	int neighborThres,
	int* surfaceParticleMark);
void ComputeScalarValues_Host(
	ZIndexGridCUDA& zgrid,
	SurfaceGridCUDA& sgrid,
	float particleSpacing,
	float infectRadius
);

void ComputeColorValues_Host(
	ZIndexGridCUDA& zgrid,
	ColorGrid& cgrid,
	float particleSpacing,
	float infectRadius
);