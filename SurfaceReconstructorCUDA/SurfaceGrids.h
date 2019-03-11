#pragma once

#include "Grids.h"

class SurfaceVertex {
public:
	int gridIndex;
	cint3 coord;
	float value;
};


class SurfaceGrid {
public:
	cfloat3 xmin;
	cfloat3 xmax;
	float cellWidth;
	cint3 cellResolution;
	cint3 vertexResolution;

	veci surfaceIndices;
	vector<SurfaceVertex> surfaceVertices;

	cfloat3 GetVertexPosition(int index) {
		return xmin + GetCoordinate(index, vertexResolution)*cellWidth;
	}
	cfloat3 GetVertexPosition(cint3 coord) {
		return xmin + coord*cellWidth;
	}
	cint3 GetVertexCoord(cfloat3 p) {
		cint3 coord;
		coord = floor((p - xmin)/cellWidth);
		return coord;
	}
	int GetSurfaceIndex(cint3 coord) {
		int gridIndex = GetIndex(coord, vertexResolution);
		if(gridIndex < surfaceIndices.size())
			return surfaceIndices[gridIndex];
		else
			return -1;
	}


	void SetSize(cfloat3 min, cfloat3 max, float cellWidth_) {
		xmin = min;
		xmax = max;
		cellWidth = cellWidth_;

		cellResolution = ceil((xmax-xmin)/cellWidth);
		xmax = xmin + cellResolution * cellWidth;
		vertexResolution = cellResolution+cint3(1, 1, 1);
		surfaceIndices.resize(vertexResolution.prod());
		surfaceVertices.clear();
		for (int i=0; i<surfaceIndices.size(); i++)
			surfaceIndices[i] = -1;

		printf("Surface Grid Resolution: %d %d %d\n", vertexResolution.x, vertexResolution.y, vertexResolution.z);
	}

	void InsertSurfaceVertex(cint3 coord) {
		int index = GetIndex(coord, vertexResolution);
		if(index == INVALID_CELL)
			return;

		if (surfaceIndices[index] == -1) {
			surfaceIndices[index] = surfaceVertices.size();
			SurfaceVertex sVertex;
			sVertex.gridIndex = index;
			sVertex.coord = coord;
			surfaceVertices.push_back(sVertex);
		}
		else 
			return;
	}
};