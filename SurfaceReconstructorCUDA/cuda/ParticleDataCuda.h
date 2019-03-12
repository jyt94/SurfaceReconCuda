#pragma once
#include "ParticleData.hpp"
#include <cuda_runtime.h>


class ParticleDataCUDA : public ParticleData {
public:
	Particle* device_particles;
	
	void LoadFromFile_CUDA(const char* filePath);

	void CopyFromDevice() {
		cudaMemcpy(particles.data(), device_particles, sizeof(Particle)*particles.size(), cudaMemcpyDeviceToHost);
	}
	void Release(){
		if (particles.size()!=0) {
			particles.clear();
			cudaFree(device_particles);
		}
	}
};
