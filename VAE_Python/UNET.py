import torch
import torch.nn as nn
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)

class UNETautoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(UNETautoencoder, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(input_size, 8)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.conv2 = DoubleConv(8, 16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = DoubleConv(16, 32)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = DoubleConv(32, 64)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.conv5 = DoubleConv(64, 128)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2,2)
        self.e_linear=nn.Linear(2048, latent_size)

        # Decoder
        self.d_linear=nn.Linear(latent_size,2048)
        self.unflatten=nn.Unflatten(1,(128,4,4))
        self.up_conv5=nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec5=DoubleConv(2*64, 64)
        self.up_conv4=nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec4=DoubleConv(2*32, 32)
        self.up_conv3=nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec3=DoubleConv(2*16, 16)
        self.up_conv2=nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.dec2=DoubleConv(2*8, 8)
        self.up_conv1=nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2)
        self.dec1=nn.Conv2d(1,1,3,1,1)
        self.out=nn.Sigmoid()

    def forward(self, x):
        # Encode
        xe1=self.conv1(x)
        xe1=self.batchnorm1(xe1)
        xp1=self.pool(xe1)

        xe2=self.conv2(xp1)
        xe2=self.batchnorm2(xe2)
        xp2=self.pool(xe2)

        xe3=self.conv3(xp2)
        xe3=self.batchnorm3(xe3)
        xp3=self.pool(xe3)

        xe4=self.conv4(xp3)
        xe4=self.batchnorm4(xe4)
        xp4=self.pool(xe4)

        xe5=self.conv5(xp4)
        xe5=self.batchnorm5(xe5)
        xp5=self.pool(xe5)
        x_e_flattened=self.flatten(xp5)
        x_encoded=self.e_linear(x_e_flattened)

        # Decode
        x_d_flattened=self.d_linear(x_encoded)
        x_unflattened=self.unflatten(x_d_flattened)

        xd5=self.up_conv5(x_unflattened)
        x_cat5 = torch.cat([xd5, xp4], dim=1)
        x=self.dec5(x_cat5)

        xd4 = self.up_conv4(x)
        x_cat4 = torch.cat([xd4, xp3], dim=1)
        x = self.dec4(x_cat4)

        xd3 = self.up_conv3(x)
        x_cat3 = torch.cat([xd3, xp2], dim=1)
        x = self.dec3(x_cat3)

        xd2 = self.up_conv2(x)
        x_cat2 = torch.cat([xd2, xp1], dim=1)
        x = self.dec2(x_cat2)

        xd1 = self.up_conv1(x)
        x=self.dec1(xd1)
        decoded=self.out(x)
        return decoded


# model=UNETautoencoder(1,128)
# print("=== ИНФОРМАЦИЯ О СЕТИ ===")
# summary(model, input_size=(1, 128, 128))  # (каналы, высота, ширина)