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

template <typename T>
T func(T x) {
	return sin(x);
}

template <typename T>
bool eq(const T a, const T b, const T epsilon) {
	if(abs(a / b - 1) < epsilon)
		return true;
	return false;
}

template <typename T>
int testGpuOnCpuMtrx(const T epsilon) { // добавить тесты транспонированных матриц.
	MtrxFactory<T> *gpuMtrxFactory = new GPUMtrxFactory<T>;
	MtrxFactory<T> *cpuMtrxFactory = new CPUMtrxFactory<T>;

	int m = 2,
		k = 3,
		n = 2,
		result = 0;

	std::vector<Mtrx<T> *> gm;
	gm.push_back(gpuMtrxFactory->create(m, k));
	gm.push_back(gpuMtrxFactory->create(k, n));
	gm.push_back(gpuMtrxFactory->create(m, n, CLEAR));
	gm.push_back(gpuMtrxFactory->create(m, n));

	std::vector<Mtrx<T> *> cm;
	cm.push_back(cpuMtrxFactory->create(m, k, CLEAR));
	cm.push_back(cpuMtrxFactory->create(k, n, CLEAR));
	cm.push_back(cpuMtrxFactory->create(m, n, CLEAR));
	cm.push_back(cpuMtrxFactory->create(m, n, CLEAR));

	delete gpuMtrxFactory;
	delete cpuMtrxFactory;

	FOR(k, 4)
		FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
				cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));

	// activation
	{
		gm[2]->activation();
		cm[2]->activation();

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -1;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// dActivation
	{
		gm[2]->dActivation();
		cm[2]->dActivation();

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -2;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// backActivation
	{
		gm[2]->backActivation();
		cm[2]->backActivation();

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -3;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// mult
	{
		gm[2]->mult(gm[0], gm[1]);
		cm[2]->mult(cm[0], cm[1]);

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -4;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// lineMult
	{
		gm[2]->lineMult(gm[3]);
		cm[2]->lineMult(cm[3]);

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -5;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// coeffMult
	{
		gm[2]->coeffMult(5);
		cm[2]->coeffMult(5);

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -6;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// lineSub
	{
		gm[2]->lineSub(gm[3]);
		cm[2]->lineSub(cm[3]);

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -7;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// transpose
	{
		gm[1]->transpose();
		cm[1]->transpose();

		FOR(i, gm[1]->getH())
			FOR(j, gm[1]->getW())
			if(!eq<T>(cm[1]->get(i, j), gm[1]->get(i, j), epsilon)) {
				result = -8;
				goto ExitTest;
			}


		gm[1]->transpose();
		cm[1]->transpose();

		FOR(i, gm[1]->getH())
			FOR(j, gm[1]->getW())
			if(!eq<T>(cm[1]->get(i, j), gm[1]->get(i, j), epsilon)) {
				result = -9;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

	// setKernel
	{
		FOR(i, 4) {
			gm[2]->set(i, i+.2f);
			cm[2]->set(i, i+.2f);
		}

		FOR(i, gm[2]->getH())
			FOR(j, gm[2]->getW())
			if(!eq<T>(cm[2]->get(i, j), gm[2]->get(i, j), epsilon)) {
				result = -10;
				goto ExitTest;
			}

		FOR(k, 4)
			FOR(i, gm[k]->getH())
			FOR(j, gm[k]->getW())
			cm[k]->set(i * gm[k]->getW() + j, gm[k]->get(i, j));
	}

ExitTest:
	FOR(i, 4) {
		delete cm[i];
		delete gm[i];
	}
	return result;
}

int main() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	// https://en.wikipedia.org/wiki/Machine_epsilon#Values_for_standard_hardware_floating_point_arithmetics
	std::cout << "test code: " << testGpuOnCpuMtrx<float>(1.19e-07f) << "\r\n";

	float trIn[TRAINSET_SIZE * INPUT_SIZE];
	float trOut[TRAINSET_SIZE * OUTPUT_SIZE];
	float x = 0, maxX = std::_Pi, num = 20, stepX = (maxX - x) / (num - 1);
	int epochs = 50, cols = 10;
	FOR(i, TRAINSET_SIZE) {
		trIn[i] = x;
		trOut[i] = func<float>(x);
		x += stepX;
	}
	int *layersSizes = new int[LAYERS_COUNT] { INPUT_SIZE, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, OUTPUT_SIZE };
	std::vector<TrainSet<float> *> trainsets;
	trainsets.push_back(new TrainSet<float>(trIn, trOut, TRAINSET_SIZE, INPUT_SIZE, OUTPUT_SIZE));

	auto t1 = std::chrono::high_resolution_clock::now();
	MtrxFactory<float> *mtrxFactory = new GPUMtrxFactory<float>;
	Neuro<float> *INN = new Neuro<float>(layersSizes, trainsets[0], mtrxFactory);
	delete mtrxFactory;
	float *MSE = new float[cols];
	FOR(i, cols) {
		FOR(j, epochs)
			INN->train(0.001f);
		MSE[i] = INN->MSE;
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	FOR(i, cols) {
		std::cout << MSE[i] << "\r\n";
	}
	delete[] MSE;
	delete[] layersSizes;
	delete INN;
	std::cout << "Done in " << duration / 1e6 << " seconds." << std::endl;
	/**/

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	if(!cuSafe(cudaDeviceReset(), "cudaDeviceReset failed!")) return -1;

	return 0;
}