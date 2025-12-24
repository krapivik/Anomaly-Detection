from dataclasses import dataclass

import torch
import torch.nn as nn
from torchsummary import summary

@dataclass
class VAEConfig:
    latent_dim: int
    input_size: int = 1
    input_shape: tuple[int, int, int] = (1, 128, 128)
    num_filter_init: int = 8
    conv_num: int = 4-1
    pool_num: int = 4


    def compute_hidden_shape(self):
        hidden_shape: tuple[int, int, int] = (self.input_shape[0]*self.num_filter_init*pow(2, self.conv_num),
                                              self.input_shape[1] // pow(2, self.pool_num),
                                              self.input_shape[2] // pow(2, self.pool_num))
        return hidden_shape

    def compute_flattened_size(self):
        hidden_shape = self.compute_hidden_shape()
        flattened_size = int(hidden_shape[0] * hidden_shape[1] * hidden_shape[2])
        return flattened_size

class Encoder(nn.Module):
    def __init__(self, configuration: VAEConfig):
        super(Encoder, self).__init__()
        self.config = configuration
        flattened_size = configuration.compute_flattened_size()

        self.conv1 = nn.Sequential(
            nn.Conv2d(configuration.input_size, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()

        self.mean_layer = nn.Linear(flattened_size, configuration.latent_dim)
        self.log_var_layer = nn.Linear(flattened_size,configuration.latent_dim)

    def forward(self, x):
        enc1 = self.conv1(x)
        pool1 = self.pool1(enc1)
        enc2 = self.conv2(pool1)
        pool2 = self.pool2(enc2)
        enc3 = self.conv3(pool2)
        pool3 = self.pool3(enc3)
        enc4 = self.conv4(pool3)
        pool4 = self.pool4(enc4)
        flat = self.flat(pool4)

        mean = self.mean_layer(flat)
        log_var = self.log_var_layer(flat)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, configuration: VAEConfig):
        super(Decoder, self).__init__()
        self.config = configuration

        flattened_size = configuration.compute_flattened_size()
        hidden_shape = configuration.compute_hidden_shape()

        self.unflat = nn.Sequential(
            nn.Linear(configuration.latent_dim, flattened_size),
            nn.Unflatten(1, hidden_shape)
        )

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(64,32,3,1,1),
            nn.ReLU(),)
        self.conv2 = nn.Sequential(
            # nn.ConvTranspose2d(32,16,2,2,0),
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 2, 2, 0, 0),
            nn.Conv2d(1, 1, 3, 1, 1),
        )
        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        unflattened = self.unflat(x)
        dec1 = self.conv1(unflattened)
        dec2 = self.conv2(dec1)
        dec3 = self.conv3(dec2)
        dec4 = self.conv4(dec3)
        decoded = self.output_layer(dec4)
        return decoded

class VAE(nn.Module):
    def __init__(self, configuration: VAEConfig):
        super(VAE, self).__init__()
        self.config = configuration
        self.encoder = Encoder(configuration)
        self.decoder = Decoder(configuration)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = VAE.reparametrisation(mean,log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

    @staticmethod
    def _log2std(log_var):
        std = torch.exp(0.5 * log_var)
        return std

    @staticmethod
    def reparametrisation(mean, log_var):
        std = VAE._log2std(log_var)
        epsilon = torch.randn_like(log_var)
        z = mean + std * epsilon
        return z

    def generate_image(self):
        z = torch.randn(1, self.config.latent_dim)
        image_tensor = self.decode(z)
        image = image_tensor.detach().cpu().numpy()
        return image

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, prediction, target, mean, log_var, num_epoch = None, kld_epoch = None):
        '''
        num_epoch - номер текущей эпохи
        kld_epoch - номер эпохи, начиная с которой усиливается влияние kld в функции потерь
        '''
        reconstruction_loss = self.mse(prediction, target)
        kld = KLDLoss.compute_kl_loss(mean, log_var)
        if kld_epoch is not None and num_epoch is not None:
            beta = min(1.0, num_epoch / kld_epoch)
            loss = reconstruction_loss + beta * kld
        else:
            beta = 1
            loss = reconstruction_loss + beta * kld
        return loss, reconstruction_loss, kld

    @staticmethod
    def compute_kl_loss(mean, log_var):
        kld = torch.mean(- 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
        return kld

# config = VAEConfig(latent_dim=128)
# model = VAE(config)
# print("=== ИНФОРМАЦИЯ О СЕТИ ===")
# summary(model, input_size=config.input_shape)  # (каналы, высота, ширина)
