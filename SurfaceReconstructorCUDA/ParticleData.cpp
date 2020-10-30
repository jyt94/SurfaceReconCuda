#include "ParticleData.hpp"

void ParticleData::LoadFromFile(const char* filePath) {
    FILE* fp = fopen(filePath, "r");
    if (fp == NULL) {
        printf("Error opening files.\n");
        return;
    }
    char buf[1024];
    fgets(buf, sizeof(buf), fp);
    printf("%s", buf);

    int vId;
    Particle p;
    float vf[3];
    int type;
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        sscanf(buf, "%d %f %f %f %d",
            &vId, &p.pos.x, &p.pos.y, &p.pos.z, &type);

        //if (type != 0)
        //	continue;
        //if (p.vf.x[0] + p.vf.x[1] == 0)
        //	continue;
        if (p.pos.x < -1.015 || p.pos.x>1.015 || p.pos.z < -1. || p.pos.z>1.)
            continue;

        if (!valid(p.pos))
            continue;

        particles.push_back(p);
    }

    fclose(fp);
}