#pragma once
#include "definitions.h"

template <typename T>
class TrainSet {
private:
	T *TrainIn = nullptr;
	T *TrainOut = nullptr;
	int trainSize = NULL;
	int inpSize = NULL;
	int outSize = NULL;
	int counter = 0;

public:
	TrainSet(T *trainIn, T *trainOut, int trainSize, int inpSize, int outSize) {
		this->trainSize = trainSize;
		this->inpSize = inpSize;
		this->outSize = outSize;
		TrainIn = new T[trainSize * inpSize];
		TrainOut = new T[trainSize * outSize];
		FOR(i, trainSize * inpSize)
			TrainIn[i] = trainIn[i];
		FOR(i, trainSize * outSize)
			TrainOut[i] = trainOut[i];
	}

	~TrainSet() {
		delete[] TrainIn;
		delete[] TrainOut;
	}

	/*
	* Метод принимает массив,
	* который заполнит текущим
	* набором для тренировки
	* (данными для входов)
	*/
	void getTrainset(T *in, T *out) {
		FOR(i, inpSize)
			in[i] = TrainIn[counter * inpSize + i];
		FOR(i, outSize)
			out[i] = TrainOut[counter * outSize + i];
	}

	void iterateTrainset() {
		if(counter >= trainSize - 1)
			counter = 0;
		else counter++;
	}

	int getTrainSize() {
		return trainSize;
	}
};