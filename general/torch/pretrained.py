import torch
from torchvision import models
from torchvision import transforms


class GenericPretrainedNetwork:
    def __init__(
        self,
        model_func=models.alexnet,
        trainable=False,
        img_size=(224, 224),
        **kwargs,
    ):
        model = model_func(pretrained=True)
        model.eval()
        self.img_size = img_size
        self.encoder = model

    @property
    def output_size(self):
        return self.encoder.output.shape[1]

    @property
    def input_shape(self):
        return self._img_shape

    def get_representation(self, samples, single=False):
        if single:
            samples = torch.expand_dims(samples, 0)
        samples_t = torch.tensor(samples, dtype=torch.float)
        rep = self.encoder.avgpool(self.encoder.features(samples_t))
        if single:
            rep = rep[0]
        return rep
