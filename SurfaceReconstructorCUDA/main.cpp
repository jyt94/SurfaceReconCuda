
#include "SurfaceReconstructor.hpp"
#include "cuda/SurfaceReconstructorCUDA.h"

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

void TestCUDA() {
	SurfaceReconstructorCUDA worker;
	worker.LoadConfig("small");

	for (int i=31; i<100; i++) {	
		char f[100];
		sprintf(f, "%03d",i);
		worker.inputFile = "particledata/"+string(f)+".txt";
		worker.outFile = "particledata/mesh/"+string(f)+".obj";
		worker.ExtractSurface();
	}
}

int main() {
	//TestParticleData();
	TestCUDA();

	return 0;
}