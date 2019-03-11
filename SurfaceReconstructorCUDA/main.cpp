
#include "SurfaceReconstructor.hpp"
#include "cuda/SurfaceReconstructorCUDA.cuh"

int main_(){
    
    SurfaceReconstructor surfaceReconstructor;

	surfaceReconstructor.LoadConfig("uniform");
    surfaceReconstructor.LoadParticle("test.txt");
	surfaceReconstructor.SetupGrids();
  
    surfaceReconstructor.ExtractSurface();
    return 0;
}

void TestParticleData() {
	SurfaceReconstructor surfaceReconstructor;

	surfaceReconstructor.LoadConfig("small");
	surfaceReconstructor.LoadParticle("particledata/000.txt");
	surfaceReconstructor.SetupGrids();

	surfaceReconstructor.ExtractSurface();
}

void Test() {
	SurfaceReconstructor sr;
	SurfaceGrid& surfaceGrid = sr.surfaceGrid;
	cfloat3 xmin = cfloat3(-0.15,-0.15,-0.15);
	cfloat3 xmax = cfloat3(0.16,0.16,0.16);
	float cellWidth = 0.01;
	surfaceGrid.SetSize(xmin, xmax, cellWidth);
	
	int numSurfaceVertices = 0;
	for(int xx=0; xx<surfaceGrid.vertexResolution.x; xx++)
		for (int yy=0; yy<surfaceGrid.vertexResolution.y; yy++)
			for (int zz=0; zz<surfaceGrid.vertexResolution.z; zz++) {
				surfaceGrid.InsertSurfaceVertex(cint3(xx,yy,zz));
			}
	printf("surface vertices: %d\n", surfaceGrid.surfaceVertices.size());
	for (auto& sv : surfaceGrid.surfaceVertices) {
		cint3 coord = sv.coord;
		cfloat3 pos = surfaceGrid.GetVertexPosition(coord);
		//sv.value = 0;
		sv.value = pos.Norm();
	}

	MarchingCube marchingCube;
	marchingCube.surfaceGrid = & surfaceGrid;
	marchingCube.SetCubeWidth(surfaceGrid.cellWidth);
	marchingCube.isoLevel = 0.1;

	marchingCube.Marching();
	marchingCube.mesh.Output("test.obj");
}

void TestCUDA() {
	SurfaceReconstructorCUDA worker;
	worker.LoadConfig("small");
	worker.inputFile = "particledata/000.txt";
	worker.outFile = "testcuda.obj";
	worker.ExtractSurface();
}

int main() {
	//main_();
	//TestParticleData();
	TestCUDA();

	return 0;
}