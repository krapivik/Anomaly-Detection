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
layers_to_featurize = ['encoder.conv1','encoder.conv2','encoder.conv3', 'encoder.conv4',
                       'encoder.mean_layer', 'encoder.log_var_layer',
                  'decoder.conv1','decoder.conv2','decoder.conv3','decoder.conv4', 'decoder.unflat']
sbs.attach_hooks(layers_to_featurize)
logits = sbs.predict(images_batch)
sbs.remove_hooks()


# print(sbs.visualization['encoder.conv1'])
means, stds =sbs.statistic_per_channel(sbs.visualization['encoder.conv4'])
# for layer in layers_to_featurize:
#     fig = sbs.visualize_outputs([layer],n_images=5)
#     layer = layer.replace('.', '_')
#     fig.savefig(f'output\\feature_maps_23122025_no_batchnorm\\{layer}.jpg')

# sbs.show_reconstruction(image)