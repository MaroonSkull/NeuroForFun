#pragma once

#define CLEAR true

constexpr int LAYERS_COUNT = 6; // перенести этот блок в данные, которые передаются в код.
constexpr int INPUT_SIZE = 1;
constexpr int OUTPUT_SIZE = 1;
constexpr int TRAINSET_SIZE = 20; // кол-во примеров для тренировки

// это фичи
#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); I++)
#define _ALIGN(N)  __declspec(align(N))

// это классика
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// для рандома
#include <random>
#include <ctime>
// для бенчмаркинга
#include <chrono>
// ну нефига себя, вот это ты даёшь...
#include <thread>
#include <mutex>
// логгирование
#include "Logger.h"
// глобальные функции
template <typename T>
T random(T low, T high);