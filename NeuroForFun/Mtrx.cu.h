// Из-за особенностей работы NVCC, запрещено
// подключать этот заголовочный файл из любых
// файлов (кроме заголовочных), не компилирующихся
// NVCC, например, можно подключать из .cu

#pragma once
#include "definitions.h"
#include "Mtrx.h"

template <typename T>
__device__ T getElement(const T *A, const size_t pitch, const int h, const int w) {
	T *pElem = (T *)((char *)A + h * pitch) + w;
	return *pElem;
}

template <typename T>
__device__ void setElement(T *A, const float param, const size_t pitch, const int h, const int w) {
	T *pElem = (T *)((char *)A + h * pitch);
	pElem[w] = param;
}

// максимум 256 потоков на блок, если clear == false
template <typename T>
__global__ void mtrxKernel(T *A, const size_t pitch, const int h, const int w, const bool clear) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if(i >= w || j >= h) return;
	if(clear)/* {
		float *p = (float *)((char *)A + j * pitch);
		p[i] = 0.0f;
	}*/
		setElement(A, 0.0f, pitch, j, i);
	else { // требует, чтобы 
		curandState state;
		curand_init(clock(), j * w + i, 0, &state);
		T rand = curand_uniform(&state) * 2 - 1; // получаем число от -1.0, до +1.0
		if(rand >= 0) {
			setElement(A, (T)(__expf(-(__powf(rand, 2.0f) / (2.0f * 0.2f))) / (sqrtf(0.2f) * 2.5f)), pitch, j, i);
		}
		else setElement(A, (T)(-expf(-(powf(rand, 2.0f) / (2.0f * 0.2f))) / (sqrtf(0.2f) * 2.5f)), pitch, j, i);
	}
}

template <typename T>
__global__ void multKernel(T *C/*, int *debug*/, const T *A, const T *B, const size_t pitchC, const size_t pitchA, const size_t pitchB, const int k) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	/*__shared__ */ T *rowA = (T *)((char *)A + j * pitchA);
	T *elemC = (T *)((char *)C + j * pitchC) + i;
	FOR(ind, k) {
		T *elemB = (T *)((char *)B + ind * pitchB) + i;
		*elemC += rowA[ind] * *elemB;
	}
}

template <typename T>
class GpuMtrx: public Mtrx<T> {
protected:
	size_t pitch = 0; // ширина строк матрицы в памяти карты
	dim3 *blocksPerGrid = nullptr;
	dim3 *threadsPerBlock = nullptr;
	cudaDeviceProp prop;

	T *A = nullptr; // указатель на память gpu
	T *B = nullptr; // указатель на память хоста
	int w;
	int h;
	bool isTransposed = false;
	bool isSynchronized = false;

	bool memorySync() {
		if(!CUSAFE(cudaMemcpy2D(B, w * sizeof(T)/*no pitch on host*/,
			A, pitch/*CUDA pitch*/,
			w * sizeof(T)/*width in bytes*/, h,
			cudaMemcpyDeviceToHost), "cudaMemcpy2D"))
				return isSynchronized; // если не вышло синхронизироваться - не меняем значение isSynchronized, нам ведь неизвестно, было ли до этого всё синхронизированно
		isSynchronized = true;
		return true;
	}

	void getDimensions() {

		// круто, когда sm <=31 штуки, но в иных ситуациях надо следить за тем, чтобы threadsPerBlock был равен или менее 1024
		// и, со временем, надо будет shared memory учитывать

		// threadsperblock <= bound

		blocksPerGrid = new dim3(w % prop.multiProcessorCount == 0 ? w / prop.multiProcessorCount : w / prop.multiProcessorCount + 1,
			h % prop.multiProcessorCount == 0 ? h / prop.multiProcessorCount : h / prop.multiProcessorCount + 1);
		threadsPerBlock = new dim3(w % blocksPerGrid->x == 0 ? w / blocksPerGrid->x : w / blocksPerGrid->x + 1,
			h % blocksPerGrid->y == 0 ? h / blocksPerGrid->y : h / blocksPerGrid->y + 1);
		std::cout << w << " x " << h << " = " << w * h << " = width, height (x x y)\n";
		std::cout << blocksPerGrid->x << " x " << blocksPerGrid->y << " = " << blocksPerGrid->x * blocksPerGrid->y << " = blocksPerGrid (x x y)\r\n";
		std::cout << threadsPerBlock->x << " x " << threadsPerBlock->y << " = " << threadsPerBlock->x * threadsPerBlock->y << " = threadsPerBlock (x x y)\r\n";
		std::cout << blocksPerGrid->x * threadsPerBlock->x << " x " << blocksPerGrid->y * threadsPerBlock->y << " = " << blocksPerGrid->x * threadsPerBlock->x * blocksPerGrid->y * threadsPerBlock->y << " = threadsPerGrid (x x y)\r\n";
		std::cout << blocksPerGrid->x * threadsPerBlock->x << " / " << blocksPerGrid->y * threadsPerBlock->y << " = " << blocksPerGrid->x * threadsPerBlock->x / static_cast<float>(blocksPerGrid->y * threadsPerBlock->y) << " = threadsPerGrid (x / y)\r\n";
		std::cout << w << " / " << h << " = " << static_cast<float>(w) / h << " = width / height\r\n";
		std::cout << pitch << " = pitch\r\n";
		std::cout << w * h * sizeof(float) / 1024.0f << " = KiB in RAM\r\n";
		std::cout << pitch * h / 1024.0f << " = KiB on GPU\r\n\r\n";

	}

	//void getDimensions(): getDimensions(1024) {};

public:
	GpuMtrx(int h, int w, bool clear) {
		this->h = h;
		this->w = w;

		B = new T[w * h];

		CUSAFE(cudaGetLastError(), "cudaGetLastError");

		CUSAFE(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties"); // По умолчанию считаем, что мы работаем с первым устройством.

		CUSAFE(cudaMallocPitch(&A, &pitch, w * sizeof(T), h), "cudaMallocPitch");

		//необходимо понять, сколько потоков нам нужно, сколько блоков
		// наша задача - загрузить все SM на полную катушку.
		// количество SM = prop.multiProcessorCount

		getDimensions();
		mtrxKernel<T> <<<*blocksPerGrid, *threadsPerBlock>>> (A, pitch, h, w, clear);

		CUSAFE(cudaGetLastError(), "mtrxKernel launch");

		CUSAFE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

		CUSAFE(cudaGetLastError(), "cudaGetLastError"); //??? Мб убрать? Выглядит излишним, вроде.
	};

	GpuMtrx(int Height, int Width): GpuMtrx(Height, Width, !CLEAR) {} // по умолчанию у нас всё заполняется случайными данными

	GpuMtrx(const Mtrx<T> &mtrx): Mtrx<T>(mtrx) {
		const GpuMtrx<T> &X = static_cast<const GpuMtrx<T>&>(mtrx);
		w = X.w;
		h = X.h;
		isTransposed = X.isTransposed;
		isSynchronized = X.isSynchronized;

		//A = new float[h * w];
		//FOR(i, w * h)
			//A[i] = X.A[i];
	};

	~GpuMtrx() {
		/*std::cout << "~Mtrx\r\n";
		cudaFree(A);
		delete A;
		delete[] B;*/
	};

	void print() {
		if(!isSynchronized)
			isSynchronized = memorySync();
		FOR(j, h) {
			FOR(i, w) {
				std::cout << B[XY(j, i)] << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	};

	// функция активации
	void activation() {
		//activationKernel
		FOR(i, w * h) {
			//A[i] = 1 / (1 + exp(-A[i]));
		}
	};

	// производная для функции акттивации
	void dActivation() { // спорный момент, но вроде, сигмоида не нужна лишний раз
	//dActivationKernel
	//float tmp;
		FOR(i, w * h) {
			//tmp = 1 / (1 + exp(-A[i]));
			//A[i] = tmp * (1 - tmp);
			//A[i] *= (1 - A[i]);
		}
	};

	// функция, получающая x из значения сигмоиды
	void backActivation() {
		//backActivationKernel
		FOR(i, w * h) {
			//A[i] = log(A[i] / (1 - A[i]));
		}
	};

	// скалярное произведение двух матриц
	void mult(Mtrx<T> *Xinp, Mtrx<T> *Yinp) {
		GpuMtrx<T> *X = static_cast<GpuMtrx<T>*>(Xinp);
		GpuMtrx<T> *Y = static_cast<GpuMtrx<T>*>(Yinp);
		if(X->w != Y->h) {
			std::cout << "error in matrix multiplication function. (width 1st and height 2nd matrix is not equal) \r\n";
			return;
		}
		if(X->h != h) {
			std::cout << "error in matrix multiplication function. (height 1st and height result matrix is not equal) \r\n";
			return;
		}
		if(Y->w != w) {
			std::cout << "error in matrix multiplication function. (width 2nd and width result matrix is not equal) \r\n";
			return;
		}

		getDimensions();
		multKernel<T> <<<*blocksPerGrid, *threadsPerBlock>>> (A, X->A, Y->A, pitch, X->getPitch(), Y->getPitch(), X->w);
		isSynchronized = false;

		CUSAFE(cudaGetLastError(), "multKernel launch");

		CUSAFE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	};

	// поэлементное умножение двух матриц
	void lineMult(Mtrx<T> *Xinp) {
		GpuMtrx<T> *X = static_cast<GpuMtrx<T>*>(Xinp);
		if((X->w != w) || (X->h != h)) {
			std::cout << "error in linear matrix multiplication function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		//lineMultKernel
		/*FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] *= X->get(i, j);*/
	};

	// умножение матрицы на коэффициент
	void coeffMult(T coeff) {
		//coeffMultKernel
		/*FOR(i, h)
			FOR(j, w)
			A[XY(i, j)] *= coeff;*/
	};

	// поэлементное вычитание из первой матрицы матрицы X
	void lineSub(Mtrx<T> *Xinp) {
		GpuMtrx<T> *X = static_cast<GpuMtrx<T>*>(Xinp);
		if((X->w != w) || (X->h != h)) {
			std::cout << "error in linear matrix addiction function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		//lineSubKernel
		/*FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] -= X->get(i, j);*/
	};

	/*
	* транспонирование напрямую не выполняется.
	* Для уменьшения работы с памятью,
	* алгоритмы работы с матрицей меняют возвращаемое значение так,
	* будто матрица отражена относительно главной оси
	*/
	void transpose() {
		//transposeKernel
		// свап значений всех переменных
		if(h + w <= 0xffff) {
			h += w;
			w = h - w;
			h -= w;
		}
		else {
			int tmp = h;
			h = w;
			w = tmp;
		}
		isTransposed = !isTransposed;
	};

	// часть хака по принудительной установке данных вместо случайных
	void set(int i, T num) {
		//setKernel
		//A[i] = num;
	};

	int getW() const {
		return w;
	};

	int getH() const {
		return h;
	};

	size_t getPitch() {
		return pitch;
	}

	T get(int h, int w) {
		if(!isSynchronized)
			isSynchronized = memorySync();
		return B[XY(h, w)];
	};

	/*
	* Функция получает координаты двумерного массива и
	* преобразовывает их в координату одномерного
	* массива, хранящегося в памяти объекта.
	*/
	int XY(int height, int width) const {
		if(!isTransposed)
			return w * height + width;
		else
			return h * width + height;
	};
};

template <typename T>
class GPUMtrxFactory: public MtrxFactory<T> {
public:
	Mtrx<T> *create(int h, int w, bool clear) {
		return new GpuMtrx<T>(h, w, clear);
	}
	Mtrx<T> *create(int Height, int Width) {
		return new GpuMtrx<T>(Height, Width);
	}
	Mtrx<T> *create(const Mtrx<T> &mtrx) {
		return new GpuMtrx<T>(mtrx);
	}
	~GPUMtrxFactory() {}
};