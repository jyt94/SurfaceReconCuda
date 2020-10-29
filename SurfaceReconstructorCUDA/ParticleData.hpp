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

    void LoadFromFile(const char* filePath){
        FILE* fp = fopen(filePath, "r");
        if(fp==NULL){
            printf("Error opening files.\n");
            return;
        }
        char buf[1024];
        fgets(buf, sizeof(buf), fp);
        printf("%s",buf);

        int vId;
        Particle p;
		float vf[3];
		int type;
        while(fgets(buf,sizeof(buf),fp)!=NULL){
            sscanf(buf, "%d %f %f %f %d",
            &vId, &p.pos.x, &p.pos.y, &p.pos.z, &type);
			
			//if (type != 0)
			//	continue;
			//if (p.vf.x[0] + p.vf.x[1] == 0)
			//	continue;
			if (p.pos.x < -1.015 || p.pos.x>1.015 || p.pos.z<-1. || p.pos.z>1.)
				continue;

			if ( !valid(p.pos) )
				continue;

            particles.push_back(p);
        }

        fclose(fp);
    }
};

