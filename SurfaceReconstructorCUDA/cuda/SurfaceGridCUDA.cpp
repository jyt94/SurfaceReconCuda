#include "SurfaceGridCUDA.h"

void SurfaceGridCUDA::SetSize(cfloat3 min, cfloat3 max, float cellWidth_) {
	xmin = min;
	xmax = max;
	cellWidth = cellWidth_;

	cellResolution = ceil((xmax-xmin)/cellWidth);
	numCells = cellResolution.prod();
	xmax = xmin + cellResolution * cellWidth;
	vertexResolution = cellResolution+cint3(1, 1, 1);

	surfaceIndices = new int[vertexResolution.prod()];
	for (int i=0; i<vertexResolution.prod(); i++)
		surfaceIndices[i] = -1;
	surfaceVertices = new vector<SurfaceVertex>();

	printf("Surface Grid Resolution: %d %d %d\n", vertexResolution.x, vertexResolution.y, vertexResolution.z);
}

void SurfaceGridCUDA::CopyToDevice() {
	cudaMemcpy(device_surfaceIndices, surfaceIndices, sizeof(int)*vertexResolution.prod(), cudaMemcpyHostToDevice);
	
	if(device_surfaceVertices!=NULL)
		cudaFree(device_surfaceVertices);
	cudaMalloc(&device_surfaceVertices, sizeof(SurfaceVertex)*numSurfaceVertices);
	cudaMemcpy(device_surfaceVertices, surfaceVertices->data(), sizeof(SurfaceVertex)*numSurfaceVertices, cudaMemcpyHostToDevice);
}

void SurfaceGridCUDA::CopyToHost() {
	cudaMemcpy(surfaceVertices->data(), device_surfaceVertices, sizeof(SurfaceVertex)*numSurfaceVertices, cudaMemcpyDeviceToHost);

}

void SurfaceGridCUDA::InsertSurfaceVertex(cint3 coord) {
	int index = GetIndex(coord, vertexResolution);
	if (index == INVALID_CELL)
		return;

	if (surfaceIndices[index] == -1) {
		surfaceIndices[index] = surfaceVertices->size();
		SurfaceVertex sVertex;
		sVertex.gridIndex = index;
		sVertex.coord = coord;
		surfaceVertices->push_back(sVertex);
	}
	else
		return;
}
