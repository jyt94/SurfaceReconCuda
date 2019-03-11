#pragma once
#include "ParticleData.hpp"
#include <cuda_runtime.h>


struct ParticleDataCUDA : ParticleData {
public:
	Particle* device_particles;
	
	void LoadFromFile_CUDA(const char* filePath) {
		LoadFromFile(filePath);
		if(particles.size()==0)
			return;
		cudaMalloc(&device_particles, sizeof(Particle)*particles.size());
		cudaMemcpy(device_particles, particles.data(), sizeof(Particle)*particles.size(), cudaMemcpyHostToDevice);
	}
	void CopyFromDevice() {
		cudaMemcpy(particles.data(), device_particles, sizeof(Particle)*particles.size(), cudaMemcpyDeviceToHost);
	}
	~ParticleDataCUDA(){
		if(particles.size()!=0)
			cudaFree(device_particles);
	}
};
