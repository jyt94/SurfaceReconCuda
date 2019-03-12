#include "ParticleDataCUDA.h"

void ParticleDataCUDA::LoadFromFile_CUDA(const char* filePath) {
	LoadFromFile(filePath);
	if (particles.size()==0)
		return;
	cudaMalloc(&device_particles, sizeof(Particle)*particles.size());
	cudaMemcpy(device_particles, particles.data(), sizeof(Particle)*particles.size(), cudaMemcpyHostToDevice);
}