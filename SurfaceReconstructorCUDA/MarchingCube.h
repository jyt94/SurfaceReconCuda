#pragma once

#include "catpaw/vec_define.h"
#include "SurfaceGrids.h"
#include <map>
#include <string>

#define OUTSIDE 99
#define NON_SURFACE 999

struct IdPoint {
	int id;
	cfloat3 pos;
};

/*
This pair<int, IdPoint> stores the edge index of the point
in its first element, and stores the point in its second element.
The attribute 'id' in IdPoint is the vertex id in the mesh,
and is not related to the edge index.
*/

typedef map<int, IdPoint> PointIdMapping;

struct Triangle {
	int pointId[3];
};

struct Mesh {
	int numVertices;
	vecf3 vertices;
	vecf3 normals;
	int numFaces;
	veci faces;

	Mesh(){}
	
	void CalculateNormals();
	void Output(string filePath);
};

struct EdgeVertex {
	int v1, v2;
};


class MarchingCube {
public:

	float isoLevel;
	SurfaceGrid* surfaceGrid;
	Mesh mesh;
	cfloat3 vertexOffsets[8];
	PointIdMapping vertexMapping;
	vector<Triangle> triangles;

	//Initialize

	void SetCubeWidth(float width);
	
	void SetIsoLevel(float isoLevel_) {
		isoLevel = isoLevel;
	}

	int GetVertexIndex(cint3 coord) {
		return GetIndex(coord, surfaceGrid->vertexResolution);
	}

	int GetEdgeIndex(cint3 coord, int edge);

	float GetValue(cint3 coord) {
		int surfaceIndex = surfaceGrid->GetSurfaceIndex(coord);
		if (surfaceIndex == -1) {
			//printf("accessing non-surface vertex\n");
			return NON_SURFACE;
		}
		else {
			return surfaceGrid->surfaceVertices[surfaceIndex].value;
		}
	}

	void Marching();
	void InsertVertex(cint3 coord, int edgeNumber, float* value);

	IdPoint CalculateIntersection(cint3 coord, int edgeNumber, float* values);

	void Reindex(PointIdMapping& vertexMapping, vector<Triangle>& triangles);
};