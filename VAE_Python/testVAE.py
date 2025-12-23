from VAE import VAE, KLDLoss,VAEConfig
from StepByStep import VAEStepByStep
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

images_batch, labels_batch = next(iter(sbs.test_loader))
sbs.attach_hooks(['encoder.conv1','encoder.conv2'])
logits = sbs.predict(images_batch)
sbs.remove_hooks()

# # fig=sbs.visualize_filters('encoder.conv2.0')
fig = sbs.visualize_outputs(['encoder.conv1'],n_images=1)
plt.show()

# sbs.show_reconstruction(image)
