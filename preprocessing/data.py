from PIL import Image, ImageOps
from torchvision import datasets, transforms


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def to_normalized_tensor(image):
    tensor = transforms.ToTensor()(image)
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(tensor)


def create_dataset(folder, training=False):
    # Получаем квадрат 256×256 без растягивания пропорций исходной фотографии.
    operations = [
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256),
    ]

    if training:
        operations.extend([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            to_normalized_tensor,
        ])
    else:
        operations.extend([
            transforms.CenterCrop(224),
            to_normalized_tensor,
        ])

    dataset = datasets.ImageFolder(
        folder, transform=transforms.Compose(operations), loader=load_rgb,
    )
    if dataset.class_to_idx != {"cats": 0, "dogs": 1}:
        raise ValueError("В папке данных должны быть два класса: cats и dogs")
    return dataset
