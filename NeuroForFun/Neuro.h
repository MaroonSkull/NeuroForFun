#pragma once
#include "definitions.h"
#include "Mtrx.h"
#include "TrainSet.h"

template <typename T>
class Neuro {
private:
	std::vector<Mtrx<T>*> layers;
	std::vector<Mtrx<T>*> weights;
	std::vector<Mtrx<T>*> deltas;
	std::vector<Mtrx<T>*> errors;
	//std::vector<int> disabled; // �������� � ���� ������ ����������� �������� ��� ������� layers, ����� ����� � ������

	T* trainIn = nullptr; // ����� ��� ���������� ��������
	T* trainOut = nullptr; // ����� ��� ���������� � ������

	//float *bias; // ����� ���-������ ������������� (float[weight.size()] �� ����, �� ������ �������� �� ����)
	// �� ����� � ������ ��������� ��� ������� � ��� �� �������.
	// ��������, ����������� � �� ������, �� ���� ����� ������������ �������� � ����������� ���������� ������?????
	TrainSet<T>* trSet = nullptr;

public:
	T MSE;
	Neuro(int* sizes, TrainSet<T>* trSet) {
		trainIn = new T[INPUT_SIZE];
		trainOut = new T[OUTPUT_SIZE];
		this->trSet = trSet;
		FOR(i, LAYERS_COUNT) // ������ ������� ��� ������������� ����������
			layers.push_back(new Mtrx<T>(sizes[i], 1, CLEAR));
		FOR(i, LAYERS_COUNT - 1) { // ������ ������� ��� �������� ����� � �� �����
			weights.push_back(new Mtrx<T>(sizes[i + 1], sizes[i]));
			deltas.push_back(new Mtrx<T>(sizes[i + 1], sizes[i], CLEAR));
			errors.push_back(new Mtrx<T>(sizes[i + 1], 1, CLEAR));
		}
		// ��� ����� ��������� ����� ���-������ ���������������� bias
		//bias = new float[sizeof(sizess)];
		//�� ������� ��� ��� ������� ����������� ����?
	}

	~Neuro() {
		delete[] trainIn;
		delete[] trainOut;
		// ��� ���� ���������� ��� ��������� �������
		FOR(i, LAYERS_COUNT)
			delete layers[i];
		FOR(i, LAYERS_COUNT - 1) {
			delete weights[i];
			delete deltas[i];
			delete errors[i];
		}
	}

	/*
	* ���������� � ��������� ������ ��������
	* ��������� ������� �������.
	*/
	Mtrx<T>* query(T* startLayer) {
		FOR(i, INPUT_SIZE)
			layers[0]->set(i, startLayer[i]);
		FOR(i, LAYERS_COUNT - 1) {
			layers[i + 1]->mult(weights[i], layers[i]);
			layers[i + 1]->activation();
		}
		return layers[LAYERS_COUNT - 1];
	}

	/*
	* ��������� ��������� � "������ �������"
	* � ����������, ����� ������ ��� ��������,
	* ����� ������ ���������� �����.
	*
	* ���������� � ������ ������ ��������
	* ��������� ��������� ������� � ��������,
	* �������� ������� ���������.
	*/
	Mtrx<T>* backQuery(T* lastLayer) {
		FOR(i, OUTPUT_SIZE)
			layers[LAYERS_COUNT]->set(i, lastLayer[i]);
		for (int i = LAYERS_COUNT; i > 0; i++) {
			layers[i - 1]->mult(weights[i], layers[i]);
			layers[i - 1]->backActivation();
		}
		return layers[0];
	}

	/*
	* ��������� �������� ����� ���� ����� ��������
	*/
	T train(T alpha) {
		MSE = static_cast<T>(0); // ������������������ ������ �� ���� ����� ��������
		FOR(iterate, TRAINSET_SIZE) {
			// ��������� ���������� ������ ������� ��� �������� � ���������� �
			trSet->getTrainset(trainIn, trainOut);
			trSet->iterateTrainset(); // � ��������� ��� ������ ����� ����� ������
			query(trainIn); // ������� ������ �������������� ��� ����
			// ������������ �������� ������ ������ � ������ ����� ������
			FOR(i, OUTPUT_SIZE) {
				errors[errors.size() - 1]->set(i, layers[LAYERS_COUNT - 1]->get(i, 0) - trainOut[i]); // -(T - O) = O - T
				MSE += pow(errors[errors.size() - 1]->get(i, 0), 2);
			}
			// ��������� ������ ��� ���� �����
			for (int i = LAYERS_COUNT - 2; i > 0; i--) {
				weights[i]->transpose();
				errors[i - 1]->mult(weights[i], errors[i]);
				weights[i]->transpose(); // ���������� �� �����
			}
			// �� ������ ������ ��������� ��������� ������ ��� ���� �����
			for (int i = LAYERS_COUNT - 1; i > 0; i--) {
				//Djk = [coeffmult]<alpha> * [linemult]<Ej*Ok(1-Ok)> * [scalarmult & transpose Oj]<OjT>
				layers[i]->dActivation();
				errors[i - 1]->lineMult(layers[i]);
				errors[i - 1]->coeffMult(alpha); // �������, ������ �� ���� ����� ����������� ����� ������������ ��������
				layers[i - 1]->transpose();
				deltas[i - 1]->mult(errors[i - 1], layers[i - 1]);
				layers[i - 1]->transpose(); // ������� �� �����
			}
			//�������� ��� ������� ����� �� ��������� ������
			FOR(i, LAYERS_COUNT - 1) {
				weights[i]->lineSub(deltas[i]);
			}
		}
		return MSE / TRAINSET_SIZE;
	}

	void print() {
		std::cout << "<PRINT>" << std::endl;
		FOR(i, LAYERS_COUNT) {
			std::cout << "-----------" << std::endl;
			std::cout << "layers[" << i << "]" << std::endl;
			layers[i]->print();
		}
		std::cout << " " << std::endl;
		FOR(i, LAYERS_COUNT - 1) {
			std::cout << "-----------" << std::endl;
			std::cout << "weights[" << i << "], deltas[" << i << "], errors[" << i << "]" << std::endl;
			weights[i]->print();
			deltas[i]->print();
			errors[i]->print();
		}
		std::cout << "</PRINT>" << std::endl;
	}
};