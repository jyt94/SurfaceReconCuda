#pragma once

#include "ParticleData.hpp"
#include "ZIndexGrid.hpp"
#include "SPHHelper.hpp"
#include "SurfaceGrids.h"

#include "MarchingCube.h"
#include "catpaw/cpXMLhelper.h"

class SurfaceReconstructor{
public:

    ParticleData particleData;
    ZIndexGrid zGrid;
    SPHHelper sphHelper;
    SurfaceGrid surfaceGrid;
    
    float particleSpacing;
    float infectRadius;
    cfloat3 padding;
	float surfaceCellWidth;
    float normThres;
    int neighborThres;
    veci surfaceParticleMark;
	float isoValue;

    
    SurfaceReconstructor(){
    }

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

    void LoadParticle(char* filePath){
        particleData.LoadFromFile(filePath);
        particleData.Analyze();
        surfaceParticleMark.resize(particleData.size());
    }

    void SetupGrids(){
		cfloat3 xmin = particleData.xmin - padding;
		cfloat3 xmax = particleData.xmax + padding;
		surfaceGrid.SetSize(xmin, xmax, particleSpacing*0.5);

		xmin = particleData.xmin - padding*2;
		xmax = particleData.xmax + padding*2;
		zGrid.SetSize(xmin, xmax, infectRadius);
    }
    
    void ExtractSurface();

    void SortParticles();
    
    void ExtractSurfaceParticles();
    void ComputeColorFieldAndMarkParticles();

    void ExtractSurfaceVertices();
	

    void ComputeScalarValues();

    void Triangulate();

    void OutputMesh(char* filePath){

    }
};