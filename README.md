# Handwriting Recognition with CNN

Проект представляет собой систему распознавания рукописных цифр на основе сверточной нейронной сети (CNN). 
Реализована как консольная часть на C++, так и интерфейсы на Python: графический и веб.

## Структура проекта

- `ConvolutionalNeuralNetwork/` — C++-реализация модели:
  - `train_model.cpp` — обучение модели.
  - `src/` и `include/` — реализация матричных операций и утилит.
  - `data/trained_model/` — сохранённые веса обученной модели.

- `Interface/` — Python-интерфейсы:
  - `modelNN/` — модель CNN на PyTorch.
  - `gui_app/` — графический интерфейс.
  - `web_app/` — веб-интерфейс (Flask).
  - `requirements.txt` — зависимости Python-проекта.

## Запуск
```bash
git clone https://github.com/zhabaa/HandwritingRecognitionCNN.git
cd HandwritingRecognitionCNN/
```
  
### C++ обучение
```bash
cd ConvolutionalNeuralNetwork
./run.sh
```

## Python: Запуск интерфейсов (Web | GUI)

### Установите зависимости:
```bash
pip install -r Interface/requirements.txt
```

### Web-приложение
```bash
python Interface/web_app/app.py
```

### GUI-приложение
```bash
python Interface/gui_app/app.py
```


## Используемые технологии

- C++: Нейронная сеть без внешних библиотек

- Python + PyTorch: Макет модели CNN

- Flask: Web-интерфейс

- PyQt: GUI приложение


## Данные
Используется классический набор MNIST с примерами изображений рукописных цифр.

## Авторы
- [@RagnarLodbrock912](https://github.com/RagnarLodbrock912) — разработка C++ ядра и базовых классов
- [@NevatusNikita](https://github.com/NevatusNikita) — разработка C++ ядра и обучения
- [@zhabaa](https://github.com/zhabaa) — разработка макета модели и web | gui интерфейсов 

