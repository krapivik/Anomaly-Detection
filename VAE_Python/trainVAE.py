import torch.optim as optim
from VAE import VAE, KLDLoss
from StepByStep import VAEStepByStep

latent_size=128
model = VAE(input_size=1,latent_size=latent_size)
loss_fn = KLDLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

sbs = VAEStepByStep(model, loss_fn=loss_fn, optimizer=optimizer)

train_data_folder = 'D:\\Rozhkova\\Projects\DL\\data\\mvtec_anomaly_detection\\screw\\train'
test_data_folder = 'D:\\Rozhkova\\Projects\\DL\\data\\mvtec_anomaly_detection\\screw\\test'
sbs.prepare_data(train_data_folder, test_data_folder, batch_size=32)
sbs.set_tensorboard('experiment')

num_epochs = 100
train_losses=sbs.train(n_epochs=num_epochs)
# sbs.save_checkpoint('checkpoints\\VAE')

# tensorboard --logdir='VAE_runs'