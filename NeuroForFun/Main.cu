#include "definitions.h"
//#include "Mtrx.cu.h"
#include "HostMtrx.h"
#include "TrainSet.h"
#include "Neuro.h"

template <typename T>
T random(T low, T high) {
	thread_local static std::random_device rd;
	thread_local static std::mt19937 rng(rd());
	thread_local std::uniform_real_distribution<> urd;
	return urd(rng, decltype(urd)::param_type{low,high});
}

bool testGpuOnCpuMtrx() {
	MtrxFactory<float> *gpuMtrxFactory = new GPUMtrxFactory<float>;
	MtrxFactory<float> *cpuMtrxFactory = new CPUMtrxFactory<float>;

	int m = 20,
		k = 3,
		n = 50;

	std::vector<Mtrx<float> *> gm;
	gm.push_back(gpuMtrxFactory->create(m, k));
	std::cout << "\r\n\r\n1\r\n\r\n";
	gm.push_back(gpuMtrxFactory->create(k, n));
	std::cout << "\r\n\r\n2\r\n\r\n";
	gm.push_back(gpuMtrxFactory->create(m, n, CLEAR));
	std::cout << "\r\n\r\n3\r\n\r\n";

	std::vector<Mtrx<float> *> cm;
	cm.push_back(cpuMtrxFactory->create(m, k, CLEAR));
	cm.push_back(cpuMtrxFactory->create(k, n, CLEAR));
	cm.push_back(cpuMtrxFactory->create(m, n, CLEAR));
	FOR(k, 2)
		FOR(i, gm[k]->getH())
		FOR(j, gm[k]->getW())
		cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));

	std::cout << "\r\n\r\n4\r\n\r\n";
	gm[2]->mult(gm[0], gm[1]);
	std::cout << "\r\n\r\n5\r\n\r\n";
	cm[2]->mult(cm[0], cm[1]);

	FOR(i, gm[2]->getH())
		FOR(j, gm[2]->getW())
			if(cm[2]->get(i, j) != gm[2]->get(i, j)) return false;
	
	return true;
}

int main() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	

	if(testGpuOnCpuMtrx()) std::cout << "passed!";
	else std::cout << "fail!";

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}

	return 0;
}