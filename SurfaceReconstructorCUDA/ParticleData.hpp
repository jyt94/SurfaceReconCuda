#pragma once

#include "catpaw/vec_define.h"
#include "catpaw/geometry_helper.h"

template <int T>
struct VecfN {
	float x[T];
};
typedef VecfN<3> Vecf3;

struct Particle{
    cfloat3 pos;
	Vecf3 vf;
};

class ParticleData{
    public:
    vector<Particle> particles;

    cfloat3 xmin;
    cfloat3 xmax;

    vector<Particle>& GetParticleArray(){
        return particles;
    }
    Particle& GetParticle(int id){
        return particles[id];
    }
    int size(){
        return particles.size();
    }

    void Analyze(){
        xmin.Set(99999,99999,99999);
        xmax.Set(-99999,-99999,-99999);
        for(int i=0; i<particles.size(); i++){
            xmin = minfilter(particles[i].pos, xmin);
            xmax = maxfilter(particles[i].pos, xmax);
        }
        
        printf3("xmin",xmin);
        printf3("xmax",xmax);
    }

	bool valid(cfloat3& p) {
		//return p.x > -0.5 && p.x < 0.5 && p.y>0 && p.y<2 && p.z>-0.5 && p.z < 0.5;
		return p.y > -1 && p.y < 2;
	}

    void LoadFromFile(const char* filePath);
};

