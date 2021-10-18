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

template <typename T>
__global__ void setKernel(T *A, const float param, const size_t pitch, const int h, const int w) {
	setElement(A, param, pitch, h, w);
}

// максимум 256 потоков на блок, если clear == false
template <typename T>
__global__ void mtrxKernel(T *A, const size_t pitch, const int h, const int w, const bool clear) {
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
	if(idx >= w || idy >= h) return;

	if(clear)
		setElement(A, static_cast<T>(0), pitch, idy, idx);
	else {
		curandState state;
		curand_init(clock(), idy * w + idx, 0, &state);
		T rand = curand_uniform(&state) * 2 - 1; // получаем число от -1.0, до +1.0
		if(rand >= 0) {
			setElement(A, __expf(-(__powf(rand, 2.0f) / (2.0f * 0.2f))) / (sqrtf(0.2f) * 2.5f), pitch, idy, idx);
		}
		else setElement(A, -expf(-(powf(rand, 2.0f) / (2.0f * 0.2f))) / (sqrtf(0.2f) * 2.5f), pitch, idy, idx);
	}
}

template <typename T>
__global__ void activationKernel(T *A, const size_t pitch, const int h, const int w) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(A, 1 / (1 + expf(-getElement(A, pitch, idy, idx))), pitch, idy, idx);
}

template <typename T>
__global__ void dActivationKernel(T *A, const size_t pitch, const int h, const int w) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(A, getElement(A, pitch, idy, idx) * (1 - getElement(A, pitch, idy, idx)), pitch, idy, idx);
}

template <typename T>
__global__ void backActivationKernel(T *A, const size_t pitch, const int h, const int w) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(A, log(getElement(A, pitch, idy, idx) / (1 - getElement(A, pitch, idy, idx))), pitch, idy, idx);
}

template <typename T> // принимает матрицы размерностей h*k и k*w
__global__ void multKernel(T *C, const T *A, const T *B, const size_t pitchC, const size_t pitchA, const size_t pitchB, const int h, const int w, const int k) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	/*__shared__ */
	T *rowA = (T *)((char *)A + idy * pitchA);
	T *elemC = (T *)((char *)C + idy * pitchC) + idx;
	*elemC = 0; // не забываем обнулять
	FOR(i, k) {
		T *elemB = (T *)((char *)B + i * pitchB) + idx;
		*elemC += rowA[i] * *elemB;
	}
}

template <typename T>
__global__ void lineMultKernel(T *A, const T *B, const size_t pitchA, const size_t pitchB, const int h, const int w) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(A, getElement(A, pitchA, idy, idx) * getElement(B, pitchB, idy, idx), pitchA, idy, idx);
}

template <typename T>
__global__ void coeffMultKernel(T *A, const T coeff, const size_t pitch, const int h, const int w) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(A, coeff * getElement(A, pitch, idy, idx), pitch, idy, idx);
}

template <typename T>
__global__ void lineSubKernel(T *A, const T *B, const size_t pitchA, const size_t pitchB, const int h, const int w) {
	const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(A, getElement(A, pitchA, idy, idx) - getElement(B, pitchB, idy, idx), pitchA, idy, idx);
}

template <typename T> // # улучшить с помощью shared memory
__global__ void transposeKernel(T *A, T *At, const size_t pitch, const size_t pitchT, const int h, const int w) {
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
	if(idx >= w || idy >= h) return;

	setElement(At, getElement(A, pitch, idy, idx), pitchT, idx, idy);
}

template <typename T>
class GpuMtrx: public Mtrx<T> {
protected:
	size_t pitch = 0; // ширина строк матрицы в памяти карты
	size_t pitchT = 0; // ширина строк транспонированной матрицы в памяти карты
	dim3 blocksPerGrid = 0;
	dim3 threadsPerBlock = 0;
	cudaDeviceProp prop;

	T *A = nullptr; // указатель на память gpu
	T *At = nullptr; // память gpu для транспонированной матрицы
	T *B = nullptr; // указатель на память хоста
	int w;
	int h;
	bool isSynchronized = false;

	void getDimensions(int bounds = 1024) {

		blocksPerGrid.x = w % prop.multiProcessorCount == 0 ? w / prop.multiProcessorCount : w / prop.multiProcessorCount + 1;
		blocksPerGrid.y = h % prop.multiProcessorCount == 0 ? h / prop.multiProcessorCount : h / prop.multiProcessorCount + 1;
		threadsPerBlock.x = w % blocksPerGrid.x == 0 ? w / blocksPerGrid.x : w / blocksPerGrid.x + 1;
		threadsPerBlock.y = h % blocksPerGrid.y == 0 ? h / blocksPerGrid.y : h / blocksPerGrid.y + 1;
		/*std::cout << w << " x " << h << " = " << w * h << " = width, height (x x y)\n";
		std::cout << blocksPerGrid->x << " x " << blocksPerGrid->y << " = " << blocksPerGrid->x * blocksPerGrid->y << " = blocksPerGrid (x x y)\r\n";
		std::cout << threadsPerBlock->x << " x " << threadsPerBlock->y << " = " << threadsPerBlock->x * threadsPerBlock->y << " = threadsPerBlock (x x y)\r\n";
		std::cout << blocksPerGrid->x * threadsPerBlock->x << " x " << blocksPerGrid->y * threadsPerBlock->y << " = " << blocksPerGrid->x * threadsPerBlock->x * blocksPerGrid->y * threadsPerBlock->y << " = threadsPerGrid (x x y)\r\n";
		std::cout << blocksPerGrid->x * threadsPerBlock->x << " / " << blocksPerGrid->y * threadsPerBlock->y << " = " << blocksPerGrid->x * threadsPerBlock->x / static_cast<float>(blocksPerGrid->y * threadsPerBlock->y) << " = threadsPerGrid (x / y)\r\n";
		std::cout << w << " / " << h << " = " << static_cast<float>(w) / h << " = width / height\r\n";
		std::cout << pitch << " = pitch\r\n";
		std::cout << w * h * sizeof(float) / 1024.0f << " = KiB in RAM\r\n";
		std::cout << pitch * h / 1024.0f << " = KiB on GPU\r\n\r\n";
		/**/
	}

public:
	GpuMtrx(int h, int w, bool clear) {
		this->h = h;
		this->w = w;

		B = new T[w * h];

		cuSafe(cudaGetLastError(), "cudaGetLastError");
		cuSafe(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties"); // По умолчанию считаем, что мы работаем с первым устройством.
		cuSafe(cudaMallocPitch(&A, &pitch, w * sizeof(T), h), "cudaMallocPitch");
		cuSafe(cudaMallocPitch(&At, &pitchT, h * sizeof(T), w), "cudaMallocPitch");

		//необходимо понять, сколько потоков нам нужно, сколько блоков
		// наша задача - загрузить все SM на полную катушку.
		// количество SM = prop.multiProcessorCount

		getDimensions();
		mtrxKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, pitch, h, w, clear);
		cuSafe(cudaGetLastError(), "mtrxKernel launch");
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения инициализации матрицы в памяти карточки
	};

	GpuMtrx(int Height, int Width): GpuMtrx(Height, Width, !CLEAR) {} // по умолчанию у нас всё заполняется случайными данными

	GpuMtrx(const Mtrx<T> &mtrx): Mtrx<T>(mtrx) { // # протестировать
		const GpuMtrx<T> &X = static_cast<const GpuMtrx<T>&>(mtrx);

		pitch = X.pitch;
		pitchT = X.pitchT;

		prop = X.prop;

		w = X.w;
		h = X.h;
		isSynchronized = false;

		B = new T[w * h];

		cuSafe(cudaMallocPitch(&A, &pitch, w * sizeof(T), h), "cudaMallocPitch");
		cuSafe(cudaMallocPitch(&At, &pitchT, h * sizeof(T), w), "cudaMallocPitch");

		FOR(i, w * h) {
			B[i] = X.B[i];
		}
		FOR(i, h) {
			FOR(j, w) {
				set(i + j, X.B[i + j]);
			}
		}
	};

	~GpuMtrx() {
		cudaFree(A);
		cudaFree(At);
		delete[] B;
	};

	void print() {
		if(!isSynchronized)
			isSynchronized = memorySync(); // # тут можно добавить проверку на успех.
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
		getDimensions();
		activationKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, pitch, h, w);
		cuSafe(cudaGetLastError(), "activationKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения активации матрицы
	};

	// производная для функции акттивации
	void dActivation() { // спорный момент, но вроде, сигмоида не нужна лишний раз
		getDimensions();
		dActivationKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, pitch, h, w);
		cuSafe(cudaGetLastError(), "dActivationKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения
	};

	// функция, получающая x из значения сигмоиды
	void backActivation() {
		getDimensions();
		backActivationKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, pitch, h, w);
		cuSafe(cudaGetLastError(), "backActivationKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения
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
		multKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, X->A, Y->A, pitch, X->getPitch(), Y->getPitch(), h, w, X->w);
		cuSafe(cudaGetLastError(), "multKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ждём, пока карточка закончит умножение
	};

	// поэлементное умножение двух матриц
	void lineMult(Mtrx<T> *Xinp) {
		GpuMtrx<T> *X = static_cast<GpuMtrx<T>*>(Xinp);
		if((X->w != w) || (X->h != h)) {
			std::cout << "error in linear matrix multiplication function. (widths & heights matrix's is not equal) \r\n";
			return;
		}

		getDimensions();
		lineMultKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, X->A, pitch, X->getPitch(), h, w);
		cuSafe(cudaGetLastError(), "lineMultKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ждём, пока карточка закончит умножение
	};

	// умножение матрицы на коэффициент
	void coeffMult(T coeff) {
		getDimensions();
		coeffMultKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, coeff, pitch, h, w);
		cuSafe(cudaGetLastError(), "coeffMultKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения
	};

	// поэлементное вычитание из первой матрицы матрицы X
	void lineSub(Mtrx<T> *Xinp) {
		GpuMtrx<T> *X = static_cast<GpuMtrx<T>*>(Xinp);
		if((X->w != w) || (X->h != h)) {
			std::cout << "error in linear matrix addiction function. (widths & heights matrix's is not equal) \r\n";
			return;
		}

		getDimensions();
		lineSubKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, X->A, pitch, X->getPitch(), h, w);
		cuSafe(cudaGetLastError(), "lineSubKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения
	};

	void transpose() {
		// честно транспонируем матрицу
		getDimensions();
		transposeKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, At, pitch, pitchT, h, w);
		cuSafe(cudaGetLastError(), "transposeKernel launch");
		{ // меняем местами значения ширины и высоты
			int tmp = h;
			h = w;
			w = tmp;
		}
		{ // выравнивания в памяти карты
			size_t tmp = pitch;
			pitch = pitchT;
			pitchT = tmp;
		}
		{ // указатели на текущий массив
			T *tmp = A;
			A = At;
			At = tmp;
		}
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения
	};

	// часть хака по принудительной установке данных вместо случайных
	void set(int i, T num) {
		getDimensions();
		setKernel<T><<<blocksPerGrid, threadsPerBlock>>>(A, num, pitch, i / w, i - (i / w) * w);
		cuSafe(cudaGetLastError(), "setKernel launch");
		isSynchronized = false;
		cuSafe(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); // ожидаем завершения
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

	T constGet(int h, int w) const {
		return B[XY(h, w)];
	}

	bool memorySync() {
		if(!cuSafe(cudaMemcpy2D(B, w * sizeof(T)/*no pitch on host*/,
			A, pitch/*CUDA pitch*/,
			w * sizeof(T)/*width in bytes*/, h,
			cudaMemcpyDeviceToHost), "cudaMemcpy2D"))
			return isSynchronized; // если не вышло синхронизироваться - не меняем значение isSynchronized, нам ведь неизвестно, было ли до этого всё синхронизированно
		isSynchronized = true;
		return true;
	}
	/*
	* Функция получает координаты двумерного массива и
	* преобразовывает их в координату одномерного
	* массива, хранящегося в памяти объекта.
	*/
	inline int XY(int height, int width) const {
		return w * height + width;
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