
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

	string wkd = "../data/";
	//wkd = "D:/jyt/Coding/Simulation/2019_1/SolverInterface - 副本/SolverInterface/results/case 0 wcsph fluid pillar/data/";
	//wkd = "D:/jyt/Coding/Simulation/2019_1/SolverInterface - 副本/SolverInterface/results/case 0 dfsph/data/";
	wkd = "D:/jyt/Coding/Simulation/2018_2 imsph/particle_data/3db data/";

	cout << worker.inputFile << endl;
	int st = 0; 
	int ed = 240;

	for (int i=st; i<ed; i++) {	
		char f[100];
		sprintf(f, "%3d",i);
		worker.inputFile = wkd+string(f)+".txt";
		
		sprintf(f, "%d", i);
		worker.outFile = wkd+"mesh/"+string(f)+".obj";
		worker.ExtractSurface();
		//worker.colorFileName = wkd + "mesh/" + string(f);
		//worker.ExtractColor();
	}
}
#include "catpaw/cpEigen.h"

int main() {
	
	//TestParticleData();
	
	TestCUDA();

	/*float t[9]={
		1,2,0, -2,1,2, 1,3,1
	};
	cmat3 a(t);
	printf("ev %f\n",eigenMax(a));*/


	return 0;
}