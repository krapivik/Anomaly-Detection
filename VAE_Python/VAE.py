import torch
import torch.nn as nn
import abc
from torchsummary import summary

class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 8, 3 ,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 3 ,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
        )

        self.mean_layer = nn.Linear(4096, latent_size)
        self.log_var_layer = nn.Linear(4096, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 4096),
            nn.Unflatten(1,(64,8,8)),
            nn.ConvTranspose2d(64,32,2,2,0),
            nn.Conv2d(32,32,3,1,1),
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
        mean, log_var = self.encode(x)
        z = self.reparametrisation(mean,log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def encode(self, x):
        encoded=self.encoder(x)
        mean, log_var = self.mean_layer(encoded), self.log_var_layer(encoded)
        return mean, log_var

    def _log2std(self,log_var):
        std = torch.exp(0.5 * log_var)
        return std

    def reparametrisation(self, mean, log_var):
        std = self._log2std(log_var)
        epsilon = torch.randn_like(log_var)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        decoded=self.decoder(z)
        return decoded

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, prediction, target, mean, log_var):
        reconstruction_loss = self.mse(prediction, target)
        beta = 1
        kld = self._compute_kl_loss(mean, log_var)
        loss = reconstruction_loss + beta * kld
        return loss, reconstruction_loss, kld

    def _compute_kl_loss(self, mean, log_var):
        kld = torch.mean(- 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
        return kld

# Создаем виртуальный класс VAE_Loss
# Создаем дочерние классы VAE_Loss под конкретные реализации функций ошибки
class AnnealingKLDLoss(nn.Module):
    def __init__(self):
        super(AnnealingKLDLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, prediction, target, mean, log_var, num_epoch):
        reconstruction_loss = self.mse(prediction, target)
        beta =  min(1.0, num_epoch / 20)
        kld = self._compute_kl_loss(mean, log_var)
        loss = reconstruction_loss + beta * kld
        return loss, reconstruction_loss, kld

    def _compute_kl_loss(self, mean, log_var):
        kld = torch.mean(- 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
        return kld

model = VAE(input_size=1, latent_size=128)
print("=== ИНФОРМАЦИЯ О СЕТИ ===")
summary(model, input_size=(1, 128, 128))  # (каналы, высота, ширина)


