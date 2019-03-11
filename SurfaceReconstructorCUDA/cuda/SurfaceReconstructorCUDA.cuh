#pragma once
#include "SurfaceReconstructor.hpp"
#include "ZIndexGrid.cuh"
#include <cuda_runtime.h>
#include "ParticleDataCuda.cuh"
#include "SurfaceGrids.h"
#include <iostream>



class SurfaceGridCUDA : public SurfaceGrid {
public:
	int* device_surfaceIndices;
	SurfaceVertex* device_surfaceVertices;

	void AllocateDeviceBuffer() {
		cudaMalloc(&device_surfaceIndices, sizeof(int)*surfaceIndices.size());
		cudaMemcpy(device_surfaceIndices, surfaceIndices.data(), sizeof(int)*surfaceIndices.size(), cudaMemcpyHostToDevice);
	}
};

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
	
	int* surfaceParticleMark;

	void LoadConfig(const char* config) {
		XMLDocument doc;
		int xmlState = doc.LoadFile("config.xml");
		Tinyxml_Reader reader;

		XMLElement* param = doc.FirstChildElement("SurfaceReconstruction")->FirstChildElement(config);
		reader.Use(param);
		particleSpacing = reader.GetFloat("particleSpacing");
		infectRadius = reader.GetFloat("infectRadius");
		float paddingx = reader.GetFloat("padding");
		padding.Set(paddingx, paddingx, paddingx);
		surfaceCellWidth = reader.GetFloat("surfaceCellWidth");

		sphHelper.SetupCubic(infectRadius*0.5);
		normThres = reader.GetFloat("normThreshold");
		neighborThres = reader.GetInt("neighborThreshold");
		isoValue = reader.GetFloat("isoValue");
	}

	void ExtractSurface() {
		LoadParticle();
		SetupGrids();
		
		ExtractSurfaceParticles();
		ExtractSurfaceVertices();
		
		ComputeScalarValues();
		Triangulate();
		
		OutputMesh();
	}

	void LoadParticle() {
		cout<<"loading particle...\n";
		particleData.LoadFromFile_CUDA(inputFile.c_str());
		particleData.Analyze();
		cudaMalloc(&surfaceParticleMark, sizeof(int)*particleData.size());
	}
	
	void SetupGrids();

	void ExtractSurfaceParticles();
	void SortParticles();
	void ComputeColorFieldAndMarkParticles();

	void ExtractSurfaceVertices();


	void ComputeScalarValues();

	void Triangulate();

	void OutputMesh();

};