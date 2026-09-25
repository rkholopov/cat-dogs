from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(pretrained=False):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
