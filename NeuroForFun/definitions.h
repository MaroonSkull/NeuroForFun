#pragma once

#define CLEAR true

constexpr int LAYERS_COUNT = 6; // ��������� ���� ���� � ������, ������� ���������� � ���.
constexpr int INPUT_SIZE = 1;
constexpr int OUTPUT_SIZE = 1;
constexpr int TRAINSET_SIZE = 20; // ���-�� �������� ��� ����������

// ��� ����
#define FOR(I,UPPERBND) for(int I = 0; I<int(UPPERBND); I++)
#define _ALIGN(N)  __declspec(align(N))

// ��� ��������
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
// ��� �������
#include <random>
#include <ctime>
// ��� ������������
#include <chrono>
// �� ������ ����, ��� ��� �� ����...
#include <thread>
#include <mutex>

// ���������� �������
template <typename T>
T random(T low, T high);