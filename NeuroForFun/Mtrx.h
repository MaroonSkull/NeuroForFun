#pragma once
#include "definitions.h"

/*template <typename T>
class Mtrx {
private:
	T* A;
	int w;
	int h;
	bool isTransposed;

	virtual T normalGaussDistribution() = 0;
	virtual void gemm_v0(Mtrx<T>* X, Mtrx<T>* Y) = 0;
	virtual void gemm_v1(Mtrx<T>* X, Mtrx<T>* Y) = 0;
public:
	virtual ~Mtrx() {};
	virtual int XY(int height, int width) = 0;
	virtual void print() = 0;
	virtual void activation() = 0;
	virtual void dActivation() = 0;
	virtual void backActivation() = 0;
	virtual void mult(Mtrx<T>* X, Mtrx<T>* Y) = 0;
	virtual void lineMult(Mtrx<T>* X) = 0;
	virtual void coeffMult(float coeff) = 0;
	virtual void lineSub(Mtrx<T>* X) = 0;
	virtual void transpose() = 0;
	virtual void set(int i, float num) = 0;
	virtual int getW() = 0;
	virtual int getH() = 0;
	virtual T get(int h, int w) = 0;
};
*/
template <typename T>
class Mtrx {
private:
	T* A;
	int w;
	int h;
	bool isTransposed = false;

	T normalGaussDistribution() {
		//float rand = (float)std::rand() / (RAND_MAX / 2) - 1; // �������� �����, ������������� (-1; 1]
		T rand = random<T>(-1, 1); // �������� �����, ������������� (-1; 1]
		T g = exp(-(pow(rand, 2) / (2 * 0.2))) / (sqrt(0.2) * 2.5); // ������ �������� �����������
		if (rand >= 0) return g;
		return -g; // ��������������, ���� ����� ���������� ���� �������������
	};

	/*
	* ��������� ��� ����������������� ������� Y
	*/
	void gemm_v0(Mtrx<T>* X, Mtrx<T>* Y) {
		FOR(i, X->h) {// i = ������ ������ �������
			FOR(j, Y->w) { // j = ������ ������ �������
				int c = XY(i, j);
				A[c] = 0;
				FOR(k, X->w) {// k = ������ ������ � ������ ������
					A[c] += X->get(i, k) * Y->get(k, j);
				}
			}
		}
	};

	void gemm_v1(Mtrx<T>* X, Mtrx<T>* Y) {
		//int M = X.h;
		//int K = X.w;
		//int N = Y.w;

		T* B = Y->A;

		FOR(i, X->h) {
			T* c = A + i * Y->w;   // ������ #i �� ������� C
			FOR(j, Y->w) c[j] = 0;
			FOR(k, X->w) {
				const T* b = B + k * Y->w; // ������ #k �� ������� B
				T a = X->get(i, k); // ������� ���� �������� �� ������ #i � ������� A
				FOR(j, Y->w) c[j] += a * b[j];
			}
		}
	};
public:
	Mtrx(int h, int w, bool clear) {
		this->h = h;
		this->w = w;
		this->A = new T[h * w];
		if (clear) FOR(i, w * h)
			this->A[i] = 0;
		else FOR(i, w * h)
			this->A[i] = normalGaussDistribution();
	};

	Mtrx(int Height, int Width) : Mtrx(Height, Width, !CLEAR) {} // �� ��������� � ��� �� ����������� ���������� �������

	Mtrx(const Mtrx<T>& mtrx) {
		this->w = mtrx.w;
		this->h = mtrx.h;
		this->isTransposed = mtrx.isTransposed;

		this->A = new T[h * w];
		FOR(i, w * h)
			this->A[i] = mtrx.A[i];
	};

	~Mtrx() {
		delete[] this->A;
	};

	/*
	* ������� �������� ���������� ���������� ������� �
	* ��������������� �� � ���������� �����������
	* �������, ����������� � ������ �������.
	*/
	int XY(int height, int width) {
		if (!isTransposed)
			return w * height + width;
		else
			return h * width + height;
	};

	void print() {
		FOR(j, h) {
			FOR(i, w) {
				std::cout << get(j, i) << "\t";
			}
			std::cout << std::endl;
		}
		//std::cout << std::endl;
	};

	// ������� ���������
	void activation() {
		FOR(i, w * h) {
			A[i] = 1 / (1 + exp(-A[i]));
		}
	};

	// ����������� ��� ������� ����������
	void dActivation() { // ������� ������, �� �����, �������� �� ����� ������ ���
	//float tmp;
		FOR(i, w * h) {
			//tmp = 1 / (1 + exp(-A[i]));
			//A[i] = tmp * (1 - tmp);
			A[i] *= (1 - A[i]);
		}
	};

	// �������, ���������� x �� �������� ��������
	void backActivation() {
		FOR(i, w * h) {
			A[i] = log(A[i] / (1 - A[i]));
		}
	};

	// ��������� ������������ ���� ������
	void mult(Mtrx<T>* X, Mtrx<T>* Y) {
		if (X->w != Y->h) {
			std::cout << "error in matrix multiplication function. (width 1st and height 2nd matrix is not equal) \r\n";
			return;
		}
		if (isTransposed or Y->isTransposed) gemm_v0(X, Y);
		else gemm_v1(X, Y);
	};

	// ������������ ��������� ���� ������
	void lineMult(Mtrx<T>* X) {
		if (X->w != w or X->h != h) {
			std::cout << "error in linear matrix multiplication function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] *= X->get(i, j);
	};

	// ��������� ������� �� �����������
	void coeffMult(float coeff) {
		FOR(i, h)
			FOR(j, w)
			A[XY(i, j)] *= coeff;
	};

	// ������������ ��������� �� ������ ������� ������� X
	void lineSub(Mtrx<T>* X) {
		if (X->w != w or X->h != h) {
			std::cout << "error in linear matrix addiction function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] -= X->get(i, j);
	};

	/*
	* ���������������� �������� �� �����������.
	* ��� ���������� ������ � �������,
	* ��������� ������ � �������� ������ ������������ �������� ���,
	* ����� ������� �������� ������������ ������� ���
	*/
	void transpose() {
		// ���� �������� ���� ����������
		if (h + w <= 0xffff) {
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

	// ����� ���� �� �������������� ��������� ������ ������ ���������
	void set(int i, float num) {
		A[i] = num;
	};

	int getW() {
		return w;
	};

	int getH() {
		return h;
	};

	T get(int h, int w) {
		return A[XY(h, w)];
	};
};

/*template <typename T>
class GPUMtrx : public Mtrx<T>
{
private:
	T* A;
	int w;
	int h;
	bool isTransposed = false;

	T normalGaussDistribution() {
		//float rand = (float)std::rand() / (RAND_MAX / 2) - 1; // �������� �����, ������������� (-1; 1]
		T rand = random<T>(-1, 1); // �������� �����, ������������� (-1; 1]
		T g = exp(-(pow(rand, 2) / (2 * 0.2))) / (sqrt(0.2) * 2.5); // ������ �������� �����������
		if (rand >= 0) return g;
		return -g; // ��������������, ���� ����� ���������� ���� �������������
	};

	/*
	* ��������� ��� ����������������� ������� Y
	*/
	/*void gemm_v0(Mtrx* X, Mtrx* Y) {
		FOR(i, X->h) {// i = ������ ������ �������
			FOR(j, Y->w) { // j = ������ ������ �������
				int c = XY(i, j);
				A[c] = 0;
				FOR(k, X->w) {// k = ������ ������ � ������ ������
					A[c] += X->get(i, k) * Y->get(k, j);
				}
			}
		}
	};

	void gemm_v1(Mtrx* X, Mtrx* Y) {
		//int M = X.h;
		//int K = X.w;
		//int N = Y.w;

		T* B = Y->A;

		FOR(i, X->h) {
			T* c = A + i * Y->w;   // ������ #i �� ������� C
			FOR(j, Y->w) c[j] = 0;
			FOR(k, X->w) {
				const T* b = B + k * Y->w; // ������ #k �� ������� B
				T a = X->get(i, k); // ������� ���� �������� �� ������ #i � ������� A
				FOR(j, Y->w) c[j] += a * b[j];
			}
		}
	};
public:
	Mtrx(int h, int w, bool clear) {
		this->h = h;
		this->w = w;
		A = new T[h * w];
		if (clear) FOR(i, w * h)
			A[i] = 0;
		else FOR(i, w * h)
			A[i] = normalGaussDistribution();
	};

	Mtrx(int Height, int Width) : Mtrx(Height, Width, !CLEAR) {} // �� ��������� � ��� �� ����������� ���������� �������

	Mtrx(const Mtrx& mtrx) {
		w = mtrx.w;
		h = mtrx.h;
		isTransposed = mtrx.isTransposed;

		A = new T[h * w];
		FOR(i, w * h)
			A[i] = mtrx.A[i];
	};

	~Mtrx() {
		delete[] A;
	};

	/*
	* ������� �������� ���������� ���������� ������� �
	* ��������������� �� � ���������� �����������
	* �������, ����������� � ������ �������.
	*/
	/*int XY(int height, int width) {
		if (!isTransposed)
			return w * height + width;
		else
			return h * width + height;
	};

	void print() {
		FOR(j, h) {
			FOR(i, w) {
				std::cout << get(j, i) << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << "poueiqe" << std::cout << std::endl;
	};

	// ������� ���������
	void activation() {
		FOR(i, w * h) {
			A[i] = 1 / (1 + exp(-A[i]));
		}
	};

	// ����������� ��� ������� ����������
	void dActivation() { // ������� ������, �� �����, �������� �� ����� ������ ���
	//float tmp;
		FOR(i, w * h) {
			//tmp = 1 / (1 + exp(-A[i]));
			//A[i] = tmp * (1 - tmp);
			A[i] *= (1 - A[i]);
		}
	};

	// �������, ���������� x �� �������� ��������
	void backActivation() {
		FOR(i, w * h) {
			A[i] = log(A[i] / (1 - A[i]));
		}
	};

	// ��������� ������������ ���� ������
	void mult(Mtrx* X, Mtrx* Y) {
		if (X->w != Y->h) {
			std::cout << "error in matrix multiplication function. (width 1st and height 2nd matrix is not equal) \r\n";
			return;
		}
		if (isTransposed or Y->isTransposed) gemm_v0(X, Y);
		else gemm_v1(X, Y);
	};

	// ������������ ��������� ���� ������
	void lineMult(Mtrx* X) {
		if (X->w != w or X->h != h) {
			std::cout << "error in linear matrix multiplication function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] *= X->get(i, j);
	};

	// ��������� ������� �� �����������
	void coeffMult(float coeff) {
		FOR(i, h)
			FOR(j, w)
			A[XY(i, j)] *= coeff;
	};

	// ������������ ��������� �� ������ ������� ������� X
	void lineSub(Mtrx* X) {
		if (X->w != w or X->h != h) {
			std::cout << "error in linear matrix addiction function. (widths & heights matrix's is not equal) \r\n";
			return;
		}
		FOR(i, X->h)
			FOR(j, X->w)
			A[XY(i, j)] -= X->get(i, j);
	};

	/*
	* ���������������� �������� �� �����������.
	* ��� ���������� ������ � �������,
	* ��������� ������ � �������� ������ ������������ �������� ���,
	* ����� ������� �������� ������������ ������� ���
	*/
	/*void transpose() {
		// ���� �������� ���� ����������
		if (h + w <= 0xffff) {
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

	// ����� ���� �� �������������� ��������� ������ ������ ���������
	void set(int i, float num) {
		A[i] = num;
	};

	int getW() {
		return w;
	};

	int getH() {
		return h;
	};

	T get(int h, int w) {
		return A[XY(h, w)];
	};
};*/

// ������� ������
/*template <typename T>
class Factory {
public:
	virtual Mtrx<T>* create() = 0;
	virtual ~Factory() {}
};

template <typename T>
class CPUMtrxFactory : public Factory<T> {
public:
	Mtrx<T>* create(int h, int w, bool clear) {
		return new CpuMtrx<T>(h, w, clear);
	}
	Mtrx<T>* create(int Height, int Width) {
		return new CpuMtrx<T>(Height, Width);
	}
	Mtrx<T>* create(const Mtrx<T>& mtrx) {
		return new CpuMtrx<T>(mtrx);
	}
	~CPUMtrxFactory() {

	}
};*/

/*template <typename T>
class GPUMtrxFactory : public Factory<T> {
public:
	Mtrx<T>* create() {
		return new GPUMtrx<T>;
	}
};*/

