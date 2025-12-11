import torch
import torch.nn as nn
import torch.optim as optim
from torch.onnx.symbolic_opset11 import unsqueeze
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from Autoencoder import Autoencoder
from UNET import UNETautoencoder
from StepByStep import StepByStep
from imageprocess import tensor_imshow

latent_size=128
model = Autoencoder(input_size=1,latent_size=latent_size)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

sbs = StepByStep(model,loss_fn=loss_fn,optimizer=optimizer)

train_data_folder='D:\\Rozhkova\\Projects\DL\\data\\mvtec_anomaly_detection\\screw\\train'
test_data_folder='D:\\Rozhkova\\Projects\\DL\\data\\mvtec_anomaly_detection\\screw\\test'
sbs.prepare_data(train_data_folder, test_data_folder, batch_size=32)
sbs.set_tensorboard('experiment')

num_epochs=80
train_losses=sbs.train(n_epochs=num_epochs)
sbs.save_checkpoint('checkpoints\\Autoencoder_20251620')

