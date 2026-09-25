import torch
from torchvision import transforms

from models.models import create_model
from preprocessing.transform import to_normalized_tensor


def prepare_image(image):
    return transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256),
        transforms.CenterCrop(224),
    ])(image)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != {"cats": 0, "dogs": 1}:
        raise ValueError("Ожидаются классы cats: 0 и dogs: 1")
    name = checkpoint["model_name"]
    model = create_model(name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, name


def predict(model, image):
    tensor = to_normalized_tensor(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    return probabilities.tolist()
