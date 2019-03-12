#pragma once

#include "MarchingCube.h"
#include "SurfaceGridCUDA.h"


class MarchingCubeCUDA : public MarchingCube {
public:

	
	SurfaceGridCUDA* surfaceGrid;
	
	//Initialize

	void SetIsoLevel(float isoLevel_) {
		isoLevel = isoLevel;
	}

	int GetVertexIndex(cint3 coord) {
		return GetIndex(coord, surfaceGrid->vertexResolution);
	}

	int GetEdgeIndex(cint3 coord, int edge);

	int GetSurfaceIndex(cint3 coord) {
		auto gridIndex = GetIndex(coord, surfaceGrid->vertexResolution);
		if(gridIndex==INVALID_CELL)
			return -1;
		return surfaceGrid->surfaceIndices[gridIndex];
	}

	float GetValue(cint3 coord) {
		auto surfaceIndex = GetSurfaceIndex(coord);
		if (surfaceIndex == -1) {
			//printf("accessing non-surface vertex\n");
			return NON_SURFACE;
		}
		else {
			return surfaceGrid->surfaceVertices->data()[surfaceIndex].value;
		}
	}

	void Marching();
	void InsertVertex(cint3 coord, int edgeNumber, float* value);

	IdPoint CalculateIntersection(cint3 coord, int edgeNumber, float* values);

	void Reindex(PointIdMapping& vertexMapping, vector<Triangle>& triangles);
};