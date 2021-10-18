# NeuroForFun
Проект, в котором я хочу реализовать конструктор произвольных нейронных сетей (perceptron, rnn(lstm etc.), CNN, GAN, transformer) и просто отточить навыки работы с C++, NVIDIA CUDA

---

Задачи:
- [ ] реализовать работу с матрицами
  - [x] на CPU
  - [x] на стандартных CUDA-ядрах
    - [ ] оптимизировать умножение
    - [ ] оптимизировать транспонирование
  - [ ] на тензорных ядрах видеокарт с compute capability 7.0+
- [x] персептроны, обратное обучение
  - [x] на CPU
  - [ ] на GPU
    - [x] CUDA cores
    - [ ] Tensor cores
- [ ] RNN
  - [ ] LSTM
- [ ] CNN
  - [ ] on CPU
  - [ ] on GPU
- [ ] GAN?
- [ ] transformers?
