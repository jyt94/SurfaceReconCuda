#pragma once

#include "catpaw/vec_define.h"

#define INVALID_CELL 999999999
#define CELL_EMPTY -1

inline HDFUNC cint3 GetCoordinate(cfloat3 pos, cfloat3 xmin, float cellWidth) {
	cint3 pCoord;
	pCoord.x = floor(pos.x - xmin.x)/cellWidth;
	pCoord.y = floor(pos.y - xmin.y)/cellWidth;
	pCoord.z = floor(pos.z - xmin.z)/cellWidth;
	return pCoord;
}

inline HDFUNC cint3 GetCoordinate(int index, cint3 resolution) {
	cint3 coord;
	coord.z = index / (resolution.x * resolution.y);
	coord.y = index % (resolution.x * resolution.y) / resolution.x;
	coord.x = index % resolution.x;
	return coord;
}

inline HDFUNC int GetIndex(cint3 coord, cint3 resolution) {
	if (coord.x<0 || coord.x>=resolution.x ||
		coord.y<0 || coord.y>=resolution.y ||
		coord.z<0 || coord.z>=resolution.z)
		return INVALID_CELL;

	return coord.z*(resolution.y*resolution.x)
		+ coord.y*resolution.x + coord.x;
}