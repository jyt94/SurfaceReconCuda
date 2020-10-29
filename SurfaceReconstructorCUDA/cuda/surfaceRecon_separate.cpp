
#include "SurfaceReconstructorCUDA.h"

void SurfaceReconstructorCUDA::ExtractColor() {
	LoadParticle();
	SetupZGrid();
	SetupColorGrids();

	SortParticles();
	ComputeColorValues();
	OutputColorValues();

	delete[] surfaceParticleMark;
	cudaFree(device_surfaceParticleMark);

	colorGrid.Release();
	zGrid.Release();
	particleData.Release();
}