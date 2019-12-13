import torch.nn as nn
from torch.nn.utils import weight_norm


class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(.2),
            weight_norm(nn.Conv2d(3, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(64, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(),

            nn.Dropout(.5),
            weight_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(128, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(),

            nn.Dropout(.5),
            weight_norm(nn.Conv2d(128, 128, 3, stride=1, padding=0)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(128, 128, 1, stride=1, padding=0)),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(128, 128, 1, stride=1, padding=0)),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        self.fc = weight_norm(nn.Linear(128, num_classes))

    def forward(self, x):
        inter_layer = self.net(x)
        logits = self.fc(inter_layer)
        return inter_layer, logits


class Generator(nn.Module):

    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(),
            Reshape((512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            weight_norm(nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = (-1,) + target_shape

    def forward(self, x):
        return x.view(self.target_shape)


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=.0, std=.05)
        nn.init.constant_(m.bias, .0)

    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0, std=.05)
