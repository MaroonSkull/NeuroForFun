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

int main() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	MtrxFactory<float> *gpuMtrxFactory = new GPUMtrxFactory<float>;

	std::vector<Mtrx<float> *> gm;
	gm.push_back(gpuMtrxFactory->create(2, 3));
	gm.push_back(gpuMtrxFactory->create(3, 2));
	gm.push_back(gpuMtrxFactory->create(2, 2, CLEAR));

	gm[0]->print();
	gm[1]->print();
	gm[2]->mult(gm[0], gm[1]);
	gm[2]->print();
	
	MtrxFactory<float> *cpuMtrxFactory = new CPUMtrxFactory<float>;

	//CpuMtrx<float> *keke = new CpuMtrx<float>(2, 3, CLEAR);

	std::vector<Mtrx<float> *> cm;
	cm.push_back(cpuMtrxFactory->create(2, 3));
	cm.push_back(cpuMtrxFactory->create(3, 2));
	cm.push_back(cpuMtrxFactory->create(2, 2, CLEAR));

	cm[0]->print();
	cm[1]->print();
	cm[2]->mult(cm[0], cm[1]);
	cm[2]->print();
	

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}

	return 0;
}