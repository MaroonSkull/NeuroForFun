#pragma once
#include "definitions.h"
#include "Mtrx.h"

template <typename T>
class CpuMtrx: public Mtrx<T> {
protected:
	T *A;
	int w;
	int h;
	bool isTransposed = false;

	T normalGaussDistribution() const {
		//float rand = (float)std::rand() / (RAND_MAX / 2) - 1; // получаем число, принадлежащее (-1; 1]
		T rand = random<T>(-1, 1); // получаем число, принадлежащее (-1; 1]
		T g = exp(-(pow(rand, static_cast<T>(2)) / (2 * 0.2))) / (sqrt(0.2) * 2.5); // делаем красивый колокольчик
		if(rand >= 0) return g;
		return -g; // переворачиваем, если число изначально было отрицательным
	};

	/*
	* Умножение при транспонированной матрице Y
	*/
	void gemm_v0(Mtrx<T> *Xinp, Mtrx<T> *Yinp) {
		CpuMtrx<T> *X = static_cast<CpuMtrx<T>*>(Xinp);
		CpuMtrx<T> *Y = static_cast<CpuMtrx<T>*>(Yinp);
		FOR(i, X->h) {// i = высота первой матрицы
			FOR(j, Y->w) { // j = ширина второй матрицы
				int c = XY(i, j);
				A[c] = 0;
				FOR(k, X->w) {// k = ширина первой и высота второй
					A[c] += X->get(i, k) * Y->get(k, j);
				}
			}
		}
	};

	void gemm_v1(Mtrx<T> *Xinp, Mtrx<T> *Yinp) {
		CpuMtrx<T> *X = static_cast<CpuMtrx<T>*>(Xinp);
		CpuMtrx<T> *Y = static_cast<CpuMtrx<T>*>(Yinp);

		//int M = X.h;
		//int K = X.w;
		//int N = Y.w;

		T *B = Y->A;

		FOR(i, X->h) {
			T *c = A + i * Y->w;   // строка #i из матрицы C
			FOR(j, Y->w) c[j] = 0;
			FOR(k, X->w) {
				const T *b = B + k * Y->w; // строка #k из матрицы B
				T a = X->get(i, k); // перебор всех значений из строки #i в матрице A
				FOR(j, Y->w) c[j] += a * b[j];
			}
		}
	};

public:
	CpuMtrx(int h, int w, bool clear) {
		this->h = h;
		this->w = w;
		A = new T[h * w];
		if(clear) FOR(i, w * h)
			A[i] = 0;
		else FOR(i, w * h)
			A[i] = normalGaussDistribution();
	};

	CpuMtrx(int Height, int Width): CpuMtrx(Height, Width, !CLEAR) {} // по умолчанию у нас всё заполняется случайными данными

	CpuMtrx(const Mtrx<T> &mtrx): Mtrx<T>(mtrx) {
		const CpuMtrx<T> &X = static_cast<const CpuMtrx<T>&>(mtrx);
		w = X.w;
		h = X.h;
		isTransposed = X.isTransposed;

		A = new T[h * w];
		FOR(i, w * h)
			A[i] = X.A[i];
	};

	~CpuMtrx() {
		delete[] A;
	};

	void print() {
		FOR(j, h) {
			FOR(i, w) {
				std::cout << get(j, i) << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	};

	// функция активации
	void activation() {
		FOR(i, w * h) {
			A[i] = 1 / (1 + exp(-A[i]));
		}
	};

	// производная для функции акттивации
	void dActivation() { // спорный момент, но вроде, сигмоида не нужна лишний раз
	//float tmp;
		FOR(i, w * h) {
			//tmp = 1 / (1 + exp(-A[i]));
			//A[i] = tmp * (1 - tmp);
			A[i] *= (1 - A[i]);
		}
	};

	// функция, получающая x из значения сигмоиды
	void backActivation() {
		FOR(i, w * h) {
			A[i] = log(A[i] / (1 - A[i]));
		}
	};

	// скалярное произведение двух матриц
	void mult(Mtrx<T> *Xinp, Mtrx<T> *Yinp) {
		CpuMtrx<T> *X = static_cast<CpuMtrx<T>*>(Xinp);
		CpuMtrx<T> *Y = static_cast<CpuMtrx<T>*>(Yinp);
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
		if(isTransposed || Y->isTransposed) gemm_v0(Xinp, Yinp);
		else gemm_v1(Xinp, Yinp);
	};

	// поэлементное умножение двух матриц
	void lineMult(Mtrx<T> *Xinp) {
		CpuMtrx<T> *X = static_cast<CpuMtrx<T>*>(Xinp);
		if((X->w != w) || (X->h != h)) {
			std::cout << "error in linear matrix multiplication function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] *= X->get(i, j);
	};

	// умножение матрицы на коэффициент
	void coeffMult(T coeff) {
		FOR(i, h)
			FOR(j, w)
			A[XY(i, j)] *= coeff;
	};

	// поэлементное вычитание из первой матрицы матрицы X
	void lineSub(Mtrx<T> *Xinp) {
		CpuMtrx<T> *X = static_cast<CpuMtrx<T>*>(Xinp);
		if((X->w != w) || (X->h != h)) {
			std::cout << "error in linear matrix addiction function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] -= X->get(i, j);
	};

	/*
	* транспонирование напрямую не выполняется.
	* Для уменьшения работы с памятью,
	* алгоритмы работы с матрицей меняют возвращаемое значение так,
	* будто матрица отражена относительно главной оси
	*/
	void transpose() {
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
		A[i] = num;
	};

	int getW() const {
		return w;
	};

	int getH() const {
		return h;
	};

	T get(int h, int w) {
		return A[XY(h, w)];
	};

	/*
	* Функция получает координаты двумерного массива и
	* преобразовывает их в координату одномерного
	* массива, хранящегося в памяти объекта.
	*/
	inline int XY(int height, int width) const {
		return !isTransposed ? w * height + width : h * width + height;
	};
};

template <typename T>
class CPUMtrxFactory: public MtrxFactory<T> {
public:
	Mtrx<T> *create(int h, int w, bool clear) {
		return new CpuMtrx<T>(h, w, clear);
	}
	Mtrx<T> *create(int Height, int Width) {
		return new CpuMtrx<T>(Height, Width);
	}
	Mtrx<T> *create(const Mtrx<T> &mtrx) {
		return new CpuMtrx<T>(mtrx);
	}
	~CPUMtrxFactory() {}
};