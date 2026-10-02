# Cats vs Dogs

Классификация фотографий кошек и собак на PyTorch: собственная CNN, предобученные AlexNet и ResNet18 с замороженными сверточными слоями. Готовые веса в репозиторий не включены.

## Quick Start

Для запуска интерфейса нужны Python 3.12, Git и один файл обученной модели. Датасет и обучение не требуются. Команды ниже — для Windows PowerShell; установка на Linux/macOS описана в разделе [«Установка»](#установка).

1. Склонируйте проект, создайте окружение и установите зависимости для CPU:

   ```powershell
   git clone https://github.com/rkholopov/cat-dogs.git
   cd cat-dogs
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   New-Item -ItemType Directory -Force artifacts
   ```

2. Скачайте `resnet18.pt` из **Assets** в [GitHub Releases](https://github.com/rkholopov/cat-dogs/releases) и положите в `artifacts`, чтобы получился путь `artifacts/resnet18.pt`. Доступность релиза и другие модели описаны в разделе [«Готовые модели»](#готовые-модели).

3. Запустите интерфейс:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run frontend/app.py
   ```

4. Откройте адрес из терминала (обычно `http://localhost:8501`), загрузите фотографию, выберите модель и нажмите **«Распознать»**.

## Структура проекта

```text
cat-dogs/
├── preprocessing/
│   ├── __init__.py
│   ├── train_test_split.py
│   └── transform.py
├── models/
│   ├── __init__.py
│   ├── models.py
│   └── resnet.py
├── train/
│   ├── __init__.py
│   └── train.py
├── test/
│   ├── __init__.py
│   ├── evaluate.py
│   └── predict.py
├── frontend/
│   ├── __init__.py
│   ├── app.py
│   └── predict.py
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── test/
│   │   ├── cats/
│   │   └── dogs/
│   └── split.csv
├── artifacts/
│   └── network1/
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
| `models/models.py` | Описывает собственную CNN и предоставляет `create_model` для выбора CNN, AlexNet или ResNet18. |
| `models/resnet.py` | Создает ResNet18 с весами ImageNet, замораживает основную сеть и заменяет `fc` на слой с двумя выходами. |
| `train/train.py` | Обучает модель, сохраняет checkpoint после каждой эпохи, ведет историю и возобновляет обучение через `--resume`. |
| `test/evaluate.py` | Загружает веса, оценивает модель на test и записывает метрики. Не обновляет параметры модели. |
| `test/predict.py` | Вычисляет вероятности кошки и собаки для новых фотографий и сохраняет их в CSV. |
| `__init__.py` | Обозначает папки с кодом как пакеты Python для импорта и запуска через `python -m`. |
| `frontend/app.py` | Интерфейс Streamlit: загрузка фотографии, показ оригинала и центральной обрезки, выбор checkpoint и результат. |
| `frontend/predict.py` | Подготавливает центральную область 224×224, загружает обученные веса и вычисляет вероятности классов. |
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
python train/train.py --model resnet18 --epochs 10 --output-dir artifacts/resnet18_run1
```

| Модель | Обучаемые слои | Learning rate |
|---|---|---|
| CNN | Все слои, с нуля | 0.001 |
| AlexNet | Все три полносвязных слоя; свертки заморожены | 0.0001 |
| ResNet18 | Только последний слой `fc`; остальная сеть заморожена | 0.001 |

AlexNet и ResNet18 загружают веса ImageNet при первом запуске; требуется интернет. Последний слой заменяется на два выхода. Все модели используют CrossEntropyLoss и AdamW с weight decay 0.0001.

В ResNet18 используется `ResNet18_Weights.IMAGENET1K_V1`. Во время обучения основная сеть остается в режиме `eval()`: статистики BatchNorm не обновляются. Общая обработка RGB 224×224 и нормализация ImageNet используются для всех моделей.

Параметры: `--model` (`cnn`, `alexnet`, `resnet18`), `--epochs`, `--resume`, `--batch-size` (по умолчанию 32), `--data-dir` (по умолчанию `data`), `--output-dir` (по умолчанию `artifacts/<название модели>`). При нехватке памяти уменьшите `--batch-size`. Для каждого нового запуска укажите новую папку результатов.

После каждой эпохи сохраняются:

- `last.pt` — веса, состояние оптимизатора, название модели, классы, номер эпохи, batch size, история и состояния генераторов случайных чисел PyTorch;
- `history.csv` — train loss и accuracy по эпохам.

**После каждой завершенной эпохи `last.pt` перезаписывается** и содержит состояние модели только на последней завершенной эпохе. Важные результаты сохраняйте отдельно: перед продолжением обучения скопируйте нужный checkpoint, например в `epoch_5.pt`, или сохраните копию всей папки запуска вместе с историей и метриками. Это позволит вернуться к выбранной версии модели, если дальнейшее обучение ухудшит результат.

## Возобновление обучения

```text
python train/train.py --resume artifacts/cnn_run1/last.pt --epochs 20
python train/train.py --resume artifacts/alexnet_run1/last.pt --epochs 20
python train/train.py --resume artifacts/resnet18_run1/last.pt --epochs 20
```

`--epochs` задает **общее число эпох**, включая уже завершенные. Если checkpoint сохранен после 10-й эпохи, команда продолжит обучение с 11-й до 20-й. Значение должно быть больше номера сохраненной эпохи.

Из checkpoint восстанавливаются модель, оптимизатор AdamW (включая learning rate), история и состояния генераторов случайных чисел PyTorch. Свертки AlexNet и ResNet18 остаются замороженными; статистики BatchNorm у ResNet18 также фиксированы. Повторно скачивать веса ImageNet для возобновления не нужно.

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
python test/evaluate.py --checkpoint artifacts/resnet18_run1/last.pt
```

Дополнительные параметры: `--data-dir` и `--batch-size`. Рядом с checkpoint записывается `metrics.json`: accuracy, macro F1, матрица ошибок, число тестовых изображений, модель и эпоха. Повторная оценка перезаписывает этот файл. Строки матрицы — настоящие классы, столбцы — предсказанные; порядок: `cats`, `dogs`.

Обучение использует только train. Валидации, early stopping и выбора лучшей эпохи нет: число эпох задается заранее, test используется для итоговой оценки моделей на одинаковых данных.

Поддерживается также запуск `python -m train.train` и `python -m test.evaluate`. В PyCharm выберите соответствующий файл, интерпретатор `.venv`, корень проекта как рабочую папку и передайте те же параметры запуска.

## Готовые модели

Файлы обученных моделей распространяются отдельно от кода через [GitHub Releases](https://github.com/rkholopov/cat-dogs/releases). Релиз `v1.0.0` подготовлен к публикации; до его публикации файлы в Assets недоступны.

После публикации скачайте из Assets хотя бы один файл: `resnet18.pt`, `alexnet.pt` или `cnn.pt`. Поместите его в папку `artifacts` в корне проекта (создайте папку, если ее нет). Распаковывать `.pt` не нужно.

| Файл | Эпохи | Test accuracy | Macro F1 |
|---|---:|---:|---:|
| `resnet18.pt` | 2 | 98,35% | 0,9835 |
| `alexnet.pt` | 10 | 96,74% | 0,9674 |
| `cnn.pt` | 30 | 90,28% | 0,9028 |

Метрики получены на одной тестовой выборке из 1995 изображений. Для первого запуска рекомендуется ResNet18.

Установите зависимости по разделу «Установка», затем запустите:

```text
python -m streamlit run frontend/app.py
```

Выберите скачанный файл в интерфейсе. Датасет и самостоятельное обучение для этого не нужны. Веса из релиза подходят для предсказания и оценки, но не для `--resume`: состояние оптимизатора в них отсутствует. Для продолжения своего обучения используйте исходный `last.pt`.

## Предсказание на новых данных

Скрипт `test/predict.py` принимает одну фотографию или папку с изображениями, включая подпапки. Поддерживаются JPG, JPEG и PNG. Разметка классов и подготовка датасета не нужны. Выполняйте команды из корня проекта в окружении с установленными зависимостями.

Для одной фотографии:

```text
python test/predict.py --checkpoint artifacts/resnet18_run1/last.pt --input "PATH_TO_IMAGE.jpg" --output artifacts/prediction.csv
```

Для папки:

```text
python test/predict.py --checkpoint artifacts/resnet18_run1/last.pt --input "PATH_TO_IMAGES" --output artifacts/predictions.csv
```

В `--checkpoint` можно указать веса CNN, AlexNet или ResNet18, включая файл из релиза, например `artifacts/resnet18.pt`. Архитектура определяется автоматически. Предсказание выполняется на CPU с той же обработкой, что в интерфейсе: RGB, масштабирование короткой стороны до 256, центральный квадрат 256×256, центральная обрезка 224×224 и нормализация ImageNet.

Вероятности выводятся в терминал и сохраняются в CSV:

| Столбец | Содержание |
|---|---|
| `image` | Абсолютный путь к фотографии. |
| `model` | Название модели: `cnn`, `alexnet` или `resnet18`. |
| `probability_cat` | Вероятность класса «кошка» от 0 до 1. |
| `probability_dog` | Вероятность класса «собака» от 0 до 1. |

Вероятности получены через softmax и в сумме равны 1 с учетом погрешности вычислений. Если `--output` не указан, результат записывается в `artifacts/predictions.csv`. Существующий файл не перезаписывается: для повторного запуска укажите новый путь. Также поддерживается запуск `python -m test.predict` с теми же параметрами.

## Интерфейс

После установки зависимостей запустите из корня проекта:

```text
python -m streamlit run frontend/app.py
```

Откройте адрес, показанный в терминале (обычно `http://localhost:8501`). Загрузите JPG или PNG, выберите checkpoint и нажмите «Распознать». Оригинал и центральная область 224×224 всегда показаны рядом после загрузки. Именно эта область после нормализации поступает в модель. Пропорции фотографии сохраняются, края обрезаются.

Приложение находит файлы `.pt` в `artifacts` и вложенных папках. Название архитектуры берется из checkpoint: поддерживаются CNN, AlexNet и ResNet18. Для запуска на другом компьютере достаточно кода, зависимостей и хотя бы одного checkpoint этого проекта, например `artifacts/alexnet_run1/last.pt`. Датасет не требуется. Веса загружаются локально; скачивание весов ImageNet не требуется.

Предсказание выполняется на CPU. Фотография обрабатывается в памяти и не записывается в датасет. Модель различает только кошек и собак; проценты не гарантируют правильный ответ для произвольных изображений. Остановить приложение можно через Ctrl+C в терминале.
