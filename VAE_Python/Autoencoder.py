import torch.nn as nn
from torchsummary import summary

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 8, 3 ,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(8, eps=1e-05),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 3 ,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=1e-05),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=1e-05),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-05),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(2048,latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.Unflatten(1,(128,4,4)),
            nn.ConvTranspose2d(128,64,2,2,0),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2, 0),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2, 0),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, 2, 0),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, 2, 0, 0),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded=self.encoder(x)
        return encoded

model = Autoencoder(input_size=1, latent_size=128)
print("=== ИНФОРМАЦИЯ О СЕТИ ===")
summary(model, input_size=(1, 128, 128))  # (каналы, высота, ширина)