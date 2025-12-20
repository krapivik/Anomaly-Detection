from VAE import VAE, KLDLoss,VAEConfig
from StepByStep import StepByStep, VAEStepByStep
import torch.optim as optim
import matplotlib.pyplot as plt

filename = 'checkpoints\\model_17122925'

config = VAEConfig(latent_dim=64)
model = VAE(config)
loss_fn = KLDLoss()
optimizer = optim.Adam(model.parameters())
sbs = VAEStepByStep(model=model, loss_fn=loss_fn, optimizer=optimizer)
sbs.load_checkpoint(filename)

train_data_folder = 'D:\\Rozhkova\\Projects\\DL\\data\\mvtec_anomaly_detection\\screw\\train'
test_data_folder = 'D:\\Rozhkova\\Projects\\DL\\data\\mvtec_anomaly_detection\\screw\\test'
sbs.prepare_data(train_data_folder, test_data_folder, batch_size=32)

# for name, module in model.named_modules():
#     print(name)


fig=sbs.visualize_filters('encoder.conv2.0')
plt.show()

# print(getattr(model,'encoder'))
# image, label = sbs.test_loader.dataset.__getitem__(80)
# image = image.unsqueeze(0)
# sbs.show_reconstruction(image)
