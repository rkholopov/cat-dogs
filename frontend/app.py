import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.predict import load_model, predict, prepare_image
from preprocessing.transform import load_rgb


@st.cache_resource(max_entries=1)
def cached_model(path, modified_ns, size):
    return load_model(path)


def main():
    st.set_page_config(page_title="Кошка или собака?", page_icon="🐾")
    st.title("Кошка или собака?")
    st.write("Загрузите фотографию питомца и выберите обученную модель.")
    st.caption("Модель выбирает только между кошкой и собакой. Проценты — оценки модели, а не гарантия правильного ответа.")

    uploaded = st.file_uploader("Фотография", type=["jpg", "jpeg", "png"])
    cropped = None
    if uploaded is not None:
        try:
            original = load_rgb(uploaded)
            cropped = prepare_image(original)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error("Не удалось открыть изображение. Выберите другую фотографию JPG или PNG.")
            st.stop()
        left, right = st.columns(2)
        left.image(original, caption="Исходная фотография", use_container_width=True)
        right.image(cropped, caption="Изображение для модели · 224 × 224", use_container_width=True)
        st.caption("Пропорции сохраняются; края обрезаются. Перед предсказанием цвета преобразуются в нормализованный тензор.")

    checkpoints = sorted((PROJECT_ROOT / "artifacts").rglob("*.pt"))
    if not checkpoints:
        st.info("Обученные веса пока не найдены. Поместите checkpoint проекта (.pt) в папку artifacts и обновите страницу.")
        st.stop()

    checkpoint = st.selectbox(
        "Модель / запуск обучения", checkpoints,
        format_func=lambda path: path.relative_to(PROJECT_ROOT / "artifacts").as_posix(),
    )
    if st.button("Распознать", type="primary", disabled=cropped is None):
        try:
            with st.spinner("Распознаём фотографию…"):
                info = checkpoint.stat()
                model, name = cached_model(str(checkpoint), info.st_mtime_ns, info.st_size)
                probabilities = predict(model, cropped)
        except Exception:
            st.error("Не удалось загрузить модель или получить предсказание. Проверьте, что выбран checkpoint этого проекта и он полностью скопирован.")
            st.stop()
        names = {"cnn": "CNN", "alexnet": "AlexNet", "resnet18": "ResNet18"}
        label = "Кошка" if probabilities[0] >= probabilities[1] else "Собака"
        st.subheader(label)
        st.caption(f"Модель: {names[name]}")
        cat, dog = st.columns(2)
        cat.metric("Кошка", f"{probabilities[0]:.1%}")
        dog.metric("Собака", f"{probabilities[1]:.1%}")


if __name__ == "__main__":
    main()
