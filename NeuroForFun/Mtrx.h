#pragma once
#include "definitions.h"

template <typename T>
class Mtrx {
protected:
	T *A = nullptr;
	int w = NULL;
	int h = NULL;
	bool isTransposed = NULL;

public:
	virtual ~Mtrx() {};
	virtual void print() = 0;
	virtual void activation() = 0;
	virtual void dActivation() = 0;
	virtual void backActivation() = 0;
	virtual void mult(Mtrx<T> *Xinp, Mtrx<T> *Yinp) = 0;
	virtual void lineMult(Mtrx<T> *Xinp) = 0;
	virtual void coeffMult(T coeff) = 0;
	virtual void lineSub(Mtrx<T> *Xinp) = 0;
	virtual void transpose() = 0;
	virtual void set(int i, T num) = 0;
	virtual int getW() const = 0;
	virtual int getH() const = 0;
	virtual T get(int h, int w) = 0;
	virtual inline int XY(int height, int width) const = 0;
};

// Фабрики матриц
template <typename T>
class MtrxFactory {
public:
	virtual Mtrx<T> *create(int h, int w, bool clear) = 0;
	virtual Mtrx<T> *create(int h, int w) = 0;
	virtual Mtrx<T> *create(const Mtrx<T> &mtrx) = 0;
	virtual ~MtrxFactory() {}
};