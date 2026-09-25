from torch import nn
from torchvision.models import AlexNet_Weights, alexnet
from models.resnet import create_resnet18


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def create_model(name, pretrained=False):
    if name == "resnet18":
        return create_resnet18(pretrained)

    if name == "cnn":
        return SmallCNN()

    if name == "alexnet":
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = alexnet(weights=weights)
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        model.classifier[6] = nn.Linear(4096, 2)
        return model

    raise ValueError(f"Неизвестная модель: {name}")
