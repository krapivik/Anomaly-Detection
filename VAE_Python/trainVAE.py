import torch.optim as optim
from VAE import VAE, KLDLoss, VAEConfig
from StepByStep import VAEStepByStep

for latent_dim in [16, 32, 64, 128]:
    config = VAEConfig(latent_dim = latent_dim)
    model = VAE(config)
    loss_fn = KLDLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    sbs = VAEStepByStep(model, loss_fn=loss_fn, optimizer=optimizer)

    train_data_folder = 'D:\\Rozhkova\\Projects\\DL\\data\\mvtec_anomaly_detection\\screw\\train'
    test_data_folder = 'D:\\Rozhkova\\Projects\\DL\\data\\mvtec_anomaly_detection\\screw\\test'
    sbs.prepare_data(train_data_folder, test_data_folder, batch_size=32)
    sbs.set_tensorboard(f"exp_lat_dim_{latent_dim}", folder = 'runs\\Bilin_Upsample')

    num_epochs = 80
    train_losses=sbs.train(n_epochs=num_epochs)

# sbs.save_checkpoint('checkpoints\\VAE')

# tensorboard --logdir='VAE_runs'