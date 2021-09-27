#include "definitions.h"
#include "Mtrx.h"
#include "TrainSet.h"
#include "Neuro.h"

std::mutex mu;

template <typename T>
T random(T low, T high) {
	thread_local static std::random_device rd;
	thread_local static std::mt19937 rng(rd());
	thread_local std::uniform_real_distribution<> urd;
	return urd(rng, decltype(urd)::param_type{ low,high });
}

template <typename T>
T func(T x) {
	return sin(x);
}

/*
* Каждый поток обучает новую нейронку с одним параметром alpha
* И собирает срез её ошибок каждые epochs шагов.
* Всего функция делает cols срезов.
*/
template <typename T>
void get(int* layersSizes, TrainSet<T>* trainset, T alpha, T epochs, int cols) {
	/*float bestAlpha, bestMSE = INFINITY, MSE;
	Neuro* bestINN = nullptr;*/
	Neuro<T>* INN = new Neuro<T>(layersSizes, trainset);
	T* MSE = new T[cols];
	auto t1 = std::chrono::high_resolution_clock::now();
	FOR(i, cols) {
		FOR(j, epochs) {
			INN->train(alpha);
		}
		MSE[i] = INN->MSE;
	}

	/*FOR(i, count) {
		alpha += step;
		Neuro* INN = new Neuro(layersSizes, trainset);
		FOR(i, 4000 - 1)
			INN->train(alpha);
		MSE = INN->train(alpha);
		if (bestMSE > MSE) {
			bestMSE = MSE;
			bestAlpha = alpha;
			delete bestINN; // удаляем предыдущую версию
			bestINN = INN;
		}
		else delete INN; // удаляем не особо успешную версию
	}*/
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

	T* in = new T[INPUT_SIZE];
	T* out = new T[OUTPUT_SIZE];


	mu.lock();
	//std::cout << std::setprecision(6) << bestMSE << std::endl;
	//std::cout << "thread #" << std::this_thread::get_id() << "\t was done in " << duration / 1e6 << "\t seconds." << std::endl;
	//std::cout << std::fixed << std::setprecision(2) << "alpha = " << bestAlpha << ", final error = " << (bestMSE * 100) << "%" << std::endl;
	std::cout << std::fixed << std::setprecision(6) << alpha << std::setprecision(2) << std::endl;
	FOR(i, cols) {
		std::cout << MSE[i] << std::endl;
	}
	std::cout << std::endl << std::endl;
	/*FOR(j, TRAINSET_SIZE) {
		trainset->getTrainset(in, out);
		trainset->iterateTrainset();
		FOR(i, INPUT_SIZE)
			std::cout << in[i] << "\t";
		std::cout << "answer: ";
		bestINN->query(in)->print();
	}
	float answer = bestINN->train(bestAlpha);
	std::cout << "MSE = " << answer << std::endl << std::endl << std::endl;
	*/
	mu.unlock();
	//delete bestINN;
	delete[] in, out;
}

int main() {
	/*float trIn[TRAINSET_SIZE * INPUT_SIZE];
	float trOut[TRAINSET_SIZE * OUTPUT_SIZE];

	float x = 0, maxX = 3, num = 20, stepX = (maxX - x) / (num - 1);

	FOR(i, TRAINSET_SIZE) {
		trIn[i] = x;
		trOut[i] = func<float>(x);
		x += stepX;
	}
	// надо придумать, как слои и их размеры настраивать
	int* layersSizes = new int[LAYERS_COUNT] { INPUT_SIZE, 10, 10, 10, 10, OUTPUT_SIZE };

	std::cout << "Available for use: " << std::thread::hardware_concurrency() << " threads" << std::endl;
	//скорость обучения где-то в районе 1/(количество нейронов)
	// мб ещё от количества синапсов зависит, чекнуть
	float alpha = 0, max = 0.008, count = std::thread::hardware_concurrency(), step = (max - alpha) / count, epochs = 1000, cols = 10; // тут вопросики

	auto t1 = std::chrono::high_resolution_clock::now();

	std::vector<std::thread> threads;
	std::vector<TrainSet<float>*> trainsets;
	FOR(i, count) {
		alpha += step;
		trainsets.push_back(new TrainSet<float>(trIn, trOut));
		threads.push_back(std::thread(get<float>, layersSizes, trainsets[i], alpha, epochs, cols)); //а результаты кто собирать будет?
	}
	FOR(i, count) threads[i].join();

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	std::cout << "done in " << duration / 1e6 << " seconds" << std::endl;
	*/

	Mtrx<double>* A = new Mtrx<double>(2, 3, CLEAR);
	Mtrx<double>* B = new Mtrx<double>(4, 3, CLEAR);
	Mtrx<double>* C = new Mtrx<double>(2, 4);

	FOR(i, A->getH() * A->getW())
		A->set(i, i);
	FOR(i, B->getH() * B->getW())
		B->set(i, i);

	B->transpose();

	std::cout << "A--------------\n";
	A->print();
	std::cout << "\nB--------------\n";
	B->print();
	std::cout << "\nC--------------\n";
	C->print();

	C->mult(A, B);

	std::cout << "\nC = A*B--------\n";
	C->print();

	return 0;
}