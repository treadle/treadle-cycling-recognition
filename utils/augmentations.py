import numpy as np

import torch

from torchvision import transforms


class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = torch.tensor(np.random.normal(loc=0, scale=self.sigma, size=x.shape)).to(x)
        x = x + noise
        return x


def init_transforms(mean, std):
    train = transforms.Compose(
        [   
            transforms.Lambda(lambda x: (x - mean) / std)
        ]
    )
    test = transforms.Compose(
        [
            transforms.Lambda(lambda x: (x - mean) / std)
        ]
    )
    return train, test