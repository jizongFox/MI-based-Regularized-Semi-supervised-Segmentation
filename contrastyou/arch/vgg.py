import torch
from torch import nn
from torch.nn import Module, Sequential
from torchvision.models import vgg11


class VGG11(Module):
    def __init__(self) -> None:
        super().__init__()
        _vgg = vgg11()
        self._features = Sequential(
            _vgg.features,
            _vgg.avgpool,
        )
        self._projection = nn.Linear(512 * 7 * 7, 256)
        self._prediction = nn.Linear(256, 256)

    def forward(self, img, return_features=False, return_projection=False, return_prediction=False):
        features = self._features(img)
        if return_features:
            return features
        projection = self._projection(torch.flatten(features, 1))
        if return_projection:
            return projection
        prediction = self._prediction(projection)
        if return_prediction:
            return prediction
        raise NotImplementedError("variables should be set to True")


class ClassifyHead(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 10),
        )


    def forward(self, features):
        return self.classifier(features)
