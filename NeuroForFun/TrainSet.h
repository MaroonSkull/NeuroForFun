#pragma once
#include "definitions.h"

template <typename T>
class TrainSet {
private:
	T* TrainIn;
	T* TrainOut;
	int trainSize;
	int hIn;
	int hOut;
	int counter = 0;

public:
	TrainSet(T* trainIn, T* trainOut) {
		TrainIn = new T[TRAINSET_SIZE * INPUT_SIZE];
		TrainOut = new T[TRAINSET_SIZE * OUTPUT_SIZE];
		FOR(i, TRAINSET_SIZE * INPUT_SIZE)
			TrainIn[i] = trainIn[i];
		FOR(i, TRAINSET_SIZE * OUTPUT_SIZE)
			TrainOut[i] = trainOut[i];
	}

	~TrainSet() {
		delete[] TrainIn;
		delete[] TrainOut;
	}

	/*
	* ћетод принимает массив,
	* который заполнит текущим
	* набором дл€ тренировки
	* (данными дл€ входов)
	*/
	void getTrainset(T* in, T* out) {
		FOR(i, INPUT_SIZE)
			in[i] = TrainIn[counter * INPUT_SIZE + i];
		FOR(i, OUTPUT_SIZE)
			out[i] = TrainOut[counter * OUTPUT_SIZE + i];
	}

	void iterateTrainset() {
		if (counter >= TRAINSET_SIZE - 1)
			counter = 0;
		else counter++;
	}

	int getTrainSize() {
		return TRAINSET_SIZE;
	}
};