# Cats vs Dogs

Классификация фотографий кошек и собак на PyTorch: собственная CNN и предобученный AlexNet с замороженными сверточными слоями. Готовые веса в репозиторий не включены.

## Структура проекта

```text
cat-dogs/
├── preprocessing/
│   ├── __init__.py
│   ├── train_test_split.py
│   └── transform.py
├── models/
│   ├── __init__.py
│   └── models.py
├── train/
│   ├── __init__.py
│   └── train.py
├── test/
│   ├── __init__.py
│   └── evaluate.py
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── test/
│   │   ├── cats/
│   │   └── dogs/
│   └── split.csv
├── artifacts/
│   └── cnn_run1/
│       ├── last.pt
│       ├── history.csv
│       └── metrics.json
├── .gitignore
├── requirements.txt
└── README.md
```

`data` появляется после подготовки датасета, `artifacts` — после запуска обучения. Названия папок экспериментов задаются через `--output-dir`; `metrics.json` создается только после оценки.

| Файл | Назначение |
|---|---|
| `preprocessing/train_test_split.py` | Определяет класс по имени JPG, копирует фотографии в train/test 80/20 и записывает состав разбиения в `split.csv`. |
| `preprocessing/transform.py` | Загружает изображения в RGB, выполняет кадрирование, аугментации и нормализацию. Создает `ImageFolder`, общий для обучения и оценки. |
| `models/models.py` | Описывает собственную CNN и создает AlexNet с двумя выходами и замороженными свертками. |
| `train/train.py` | Обучает модель, сохраняет checkpoint после каждой эпохи, ведет историю и возобновляет обучение через `--resume`. |
| `test/evaluate.py` | Загружает веса, оценивает модель на test и записывает метрики. Не обновляет параметры модели. |
| `__init__.py` | Обозначает папки с кодом как пакеты Python для импорта и запуска через `python -m`. |
| `requirements.txt` | Фиксирует версии библиотек. |
| `.gitignore` | Исключает из Git данные, результаты, окружение и локальные настройки IDE. |
| `README.md` | Содержит инструкции по установке и использованию проекта. |

Папки `train` и `test` содержат код, а `data/train` и `data/test` — изображения. В `artifacts` для каждого эксперимента создается отдельная папка. Локальные `.venv`, `data`, `artifacts` и `.idea` не включаются в Git.

## Установка

Требуется 64-битный Python 3.9–3.12. В примерах используется Python 3.12. Склонируйте репозиторий и выполняйте команды из его корня.

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
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Для NVIDIA GPU установите совместимую с драйвером CUDA-сборку torch 2.8.0 и torchvision 0.23.0 по [инструкции PyTorch](https://pytorch.org/get-started/locally/), затем выполните `python -m pip install -r requirements.txt`. На macOS используйте `python -m pip install -r requirements.txt`.

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

Параметры: `--model`, `--epochs`, `--resume`, `--batch-size` (по умолчанию 32), `--data-dir` (по умолчанию `data`), `--output-dir` (по умолчанию `artifacts/cnn` или `artifacts/alexnet`). При нехватке памяти уменьшите `--batch-size`. Для каждого нового запуска укажите новую папку результатов.

После каждой эпохи сохраняются:

- `last.pt` — веса, состояние оптимизатора, название модели, классы, номер эпохи, batch size, история и состояния генераторов случайных чисел PyTorch;
- `history.csv` — train loss и accuracy по эпохам.

## Возобновление обучения

```text
python train/train.py --resume artifacts/cnn_run1/last.pt --epochs 20
python train/train.py --resume artifacts/alexnet_run1/last.pt --epochs 20
```

`--epochs` задает **общее число эпох**, включая уже завершенные. Если checkpoint сохранен после 10-й эпохи, команда продолжит обучение с 11-й до 20-й. Значение должно быть больше номера сохраненной эпохи.

Из checkpoint восстанавливаются модель, оптимизатор AdamW (включая learning rate), история и состояния генераторов случайных чисел PyTorch. Свертки AlexNet остаются замороженными. Повторно скачивать веса ImageNet для возобновления не нужно.

`--model` указывать не требуется: название берется из checkpoint. Если указать другую модель, запуск завершится с ошибкой. Batch size также восстанавливается; его можно переопределить через `--batch-size`, например при нехватке памяти.

По умолчанию обновляются `last.pt` и `history.csv` в папке исходного checkpoint. История предыдущих эпох сохраняется. Чтобы продолжить в отдельной новой папке:

```text
python train/train.py --resume artifacts/cnn_run1/last.pt --epochs 20 --output-dir artifacts/cnn_run2
```

Checkpoint записывается после завершения эпохи. При остановке посреди эпохи ее прогресс не сохраняется: после возобновления она будет выполнена заново. Если первая эпоха не завершилась, checkpoint еще нет.

Для продолжения на другом компьютере перенесите папку запуска из `artifacts`, используйте ту же версию кода и те же данные, создайте новое окружение и установите зависимости. Другой путь к данным задается через `--data-dir`. Переносить `.venv` не нужно. Полное совпадение результатов при смене оборудования, библиотек или batch size не гарантируется.

Старые checkpoints, содержащие только веса, подходят для оценки, но не поддерживают `--resume`: состояние оптимизатора в них отсутствует. Новые checkpoints сохраняют все необходимое автоматически. После дополнительного обучения повторно запустите оценку, чтобы обновить `metrics.json`.

## Оценка

```text
python test/evaluate.py --checkpoint artifacts/cnn_run1/last.pt
python test/evaluate.py --checkpoint artifacts/alexnet_run1/last.pt
```

Дополнительные параметры: `--data-dir` и `--batch-size`. Рядом с checkpoint записывается `metrics.json`: accuracy, macro F1, матрица ошибок, число тестовых изображений, модель и эпоха. Повторная оценка перезаписывает этот файл. Строки матрицы — настоящие классы, столбцы — предсказанные; порядок: `cats`, `dogs`.

Обучение использует только train. Валидации, early stopping и выбора лучшей эпохи нет: число эпох задается заранее, test используется для итоговой оценки обеих моделей на одинаковых данных.

Поддерживается также запуск `python -m train.train` и `python -m test.evaluate`. В PyCharm выберите соответствующий файл, интерпретатор `.venv`, корень проекта как рабочую папку и передайте те же параметры запуска.
