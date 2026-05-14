from torch import nn
import torchvision

def resnet18_model(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
model = resnet18_model(num_classes=10)
