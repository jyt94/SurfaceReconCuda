
#pragma once

#include <cuda_runtime.h>
#include <math.h>

inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline void computeBlockSize(int n, int blockSize, int &numBlocks, int &numThreads)
{
	numThreads = fmin(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}