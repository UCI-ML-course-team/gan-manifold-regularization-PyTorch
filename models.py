import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Dropout(.2),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(96, 192, 3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Dropout(.2),
            nn.Conv2d(192, 192, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(192, 192, 1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(192, 192, 1, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.MaxPool2d(4, stride=1),
            Flatten()
        )

        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        features = self.net(x)
        logits = self.fc(features)
        return features, logits


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
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
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
        nn.init.normal_(m.weight, mean=.0, std=.1)
        nn.init.constant_(m.bias, .0)

    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0, std=.05)
