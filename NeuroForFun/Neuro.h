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
	//std::vector<int> disabled; // содержит в себе номера отключенных нейронов для каждого layers, кроме входа и выхода

	T* trainIn = nullptr; // нужно для заполнения нейронки
	T* trainOut = nullptr; // нужно для вычисления её ошибки

	//float *bias; // потом как-нибудь задействовать (float[weight.size()] по идее, по одному значению на слой)
	// но можно и просто постоянно его хранить в тех же веигхтс.
	// возможно, потребность в нём отпадёт, но пока делаю классчиеский алгоритм с полноценным умножением матриц?????
	TrainSet<T>* trSet = nullptr;

public:
	T MSE;
	Neuro(int* sizes, TrainSet<T>* trSet, MtrxFactory<T>* mtrxFactory) {
		trainIn = new T[INPUT_SIZE];
		trainOut = new T[OUTPUT_SIZE];
		this->trSet = trSet;

		FOR(i, LAYERS_COUNT) // создаём матрицы для промежуточных вычислений
			layers.push_back(mtrxFactory->create(sizes[i], 1, CLEAR));
		FOR(i, LAYERS_COUNT - 1) { // создаём матрицы для хранения весов и их дельт
			weights.push_back(mtrxFactory->create(sizes[i + 1], sizes[i]));
			deltas.push_back(mtrxFactory->create(sizes[i + 1], sizes[i], CLEAR));
			errors.push_back(mtrxFactory->create(sizes[i + 1], 1, CLEAR));
		}
		// тут нужно нормально потом как-нибудь инициализировать bias
		//bias = new float[sizeof(sizess)];
		//мб хранить вес для каждого конкретного слоя?
	}

	~Neuro() {
		delete[] trainIn;
		delete[] trainOut;
		// тут надо освободить все созданные матрицы
		FOR(i, LAYERS_COUNT)
			delete layers[i];
		FOR(i, LAYERS_COUNT - 1) {
			delete weights[i];
			delete deltas[i];
			delete errors[i];
		}
	}

	/*
	* Записывает в последний вектор нейронки
	* результат прямого прохода.
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
	* Позволяет заглянуть в "чёрную коробку"
	* и посмотреть, какой вопрос ждёт нейронка,
	* чтобы выдать правильный ответ.
	*
	* Записывает в первый вектор нейронки
	* результат обратного прохода с функцией,
	* обратной функции активации.
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
	* Прогоняет нейронку через одну эпоху обучения
	*/
	T train(T alpha) {
		MSE = static_cast<T>(0); // среднеквадратичная ошибка за одну эпоху обучения
		FOR(iterate, TRAINSET_SIZE) {
			// заполняем нейроночку первым набором для обучения и опрашиваем её
			trSet->getTrainset(trainIn, trainOut);
			trSet->iterateTrainset(); // в следующий раз возьмём новый набор данных
			query(trainIn); // функция просто инициализирует все слои
			// рассчитываем величину ошибки выхода и ошибки всего ответа
			FOR(i, OUTPUT_SIZE) {
				errors[errors.size() - 1]->set(i, layers[LAYERS_COUNT - 1]->get(i, 0) - trainOut[i]); // -(T - O) = O - T
				MSE += pow(errors[errors.size() - 1]->get(i, 0), 2);
			}
			// Вычисляем ошибки для всех весов
			for (int i = LAYERS_COUNT - 2; i > 0; i--) {
				weights[i]->transpose();
				errors[i - 1]->mult(weights[i], errors[i]);
				weights[i]->transpose(); // исправляем за собой
			}
			// на основе ошибок вычисляем требуемые дельты для всех весов
			for (int i = LAYERS_COUNT - 1; i > 0; i--) {
				//Djk = [coeffmult]<alpha> * [linemult]<Ej*Ok(1-Ok)> * [scalarmult & transpose Oj]<OjT>
				layers[i]->dActivation();
				errors[i - 1]->lineMult(layers[i]);
				errors[i - 1]->coeffMult(alpha); // кажется, именно на этом этапе эффективнее всего использовать скорость
				layers[i - 1]->transpose();
				deltas[i - 1]->mult(errors[i - 1], layers[i - 1]);
				layers[i - 1]->transpose(); // подмети за собой
			}
			//изменяем все матрицы весов на найденные дельты
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