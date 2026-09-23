# Cats vs Dogs

Классификация фотографий кошек и собак на PyTorch: собственная CNN и предобученный AlexNet с замороженными сверточными слоями. Готовые веса в репозиторий не включены.

## Структура

```text
preprocessing/
    train_test_split.py
    data.py
models/
    models.py
train/
    train.py
test/
    evaluate.py
requirements.txt
```

`preprocessing` содержит разбиение и загрузку данных, `models` — архитектуры моделей, `train` — обучение, `test` — оценку. Локальные папки `.venv`, `data` и `artifacts` исключены из Git.

## Установка

Требуется 64-битный Python 3.12. Склонируйте репозиторий и выполняйте команды из его корня.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Для CPU на Windows/Linux:

```text
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Для NVIDIA GPU установите совместимую с драйвером CUDA-сборку torch 2.14.0 и torchvision 0.29.0 по [инструкции PyTorch](https://pytorch.org/get-started/locally/), затем выполните `python -m pip install -r requirements.txt`. На macOS используйте `python -m pip install -r requirements.txt`.

Устройство выбирается автоматически: CUDA при наличии поддержки, иначе CPU. Ускорение MPS не используется. Проверка установки:

```text
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Данные

[Скачать датасет](https://disk.yandex.ru/d/VzV5KXL8Y7Vvgg). Распакуйте фотографии и передайте путь к папке с файлами `cat.….jpg` и `dog.….jpg`:

```text
python preprocessing/train_test_split.py --source "PATH_TO_IMAGES" --dry-run
python preprocessing/train_test_split.py --source "PATH_TO_IMAGES"
```

Первая команда показывает план, вторая копирует файлы. Разбиение — 80/20 отдельно для каждого класса, seed — 42. Исходники не изменяются; существующая выходная папка не перезаписывается. Путь назначения можно задать через `--output`.

```text
data/
    train/
        cats/
        dogs/
    test/
        cats/
        dogs/
    split.csv
```

`split.csv` хранит состав выборок. Подготовленную папку `data` можно перенести целиком на другой компьютер без повторного разбиения.

Изображения преобразуются при загрузке: RGB, масштабирование короткой стороны до 256, центральный квадрат 256×256. Для train используются случайная область 224×224 и горизонтальное отражение с вероятностью 50%; для test — центр 224×224. Нормализация: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`. Файлы на диске не меняются.

## Обучение

```text
python train/train.py --model cnn --epochs 10 --output-dir artifacts/cnn_run1
python train/train.py --model alexnet --epochs 10 --output-dir artifacts/alexnet_run1
```

| Модель | Обучаемые слои | Learning rate |
|---|---|---|
| CNN | Все слои, с нуля | 0.001 |
| AlexNet | Все три полносвязных слоя; свертки заморожены | 0.0001 |

AlexNet загружает веса ImageNet при первом запуске; требуется интернет. Последний слой заменяется на два выхода. Обе модели используют CrossEntropyLoss и AdamW с weight decay 0.0001.

Параметры: `--model`, `--epochs`, `--batch-size` (по умолчанию 32), `--data-dir` (по умолчанию `data`), `--output-dir` (по умолчанию `artifacts/cnn` или `artifacts/alexnet`). При нехватке памяти уменьшите `--batch-size`. Для каждого нового запуска укажите новую папку результатов.

После каждой эпохи сохраняются:

- `last.pt` — веса, название модели, классы и номер эпохи;
- `history.csv` — train loss и accuracy по эпохам.

Возобновление обучения не реализовано. Для переноса результатов сохраните папку запуска из `artifacts`. На другом компьютере создайте новое окружение и установите зависимости; `.venv` переносить не нужно.

## Оценка

```text
python test/evaluate.py --checkpoint artifacts/cnn_run1/last.pt
python test/evaluate.py --checkpoint artifacts/alexnet_run1/last.pt
```

Дополнительные параметры: `--data-dir` и `--batch-size`. Рядом с checkpoint записывается `metrics.json`: accuracy, macro F1, матрица ошибок, число тестовых изображений, модель и эпоха. Повторная оценка перезаписывает этот файл. Строки матрицы — настоящие классы, столбцы — предсказанные; порядок: `cats`, `dogs`.

Обучение использует только train. Валидации, early stopping и выбора лучшей эпохи нет: число эпох задается заранее, test используется для итоговой оценки обеих моделей на одинаковых данных.

Поддерживается также запуск `python -m train.train` и `python -m test.evaluate`. В PyCharm выберите соответствующий файл, интерпретатор `.venv`, корень проекта как рабочую папку и передайте те же параметры запуска.
