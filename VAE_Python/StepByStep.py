import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        # Here we define the attributes of our class
        # We start by storing the arguments as attributes
        # to use later
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        self.train_loader = None
        self.test_loader = None
        self.writer = None

        # Variables
        self.losses = []
        self.total_epochs = 0

        # Functions
        self.train_step_fn = self._make_train_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Couldn't send it to {device}, / sending it to {self.device} instead.")
        self.model.to(self.device)

    def _set_loaders(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def prepare_data(self,train_data_folder_pass, test_data_folder_pass, batch_size):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # явно указываем 1 канал
            transforms.Resize((128, 128)),  # изменяем размер
            transforms.ToTensor()])

        train_dataset = datasets.ImageFolder(root=train_data_folder_pass, transform=transform)

        test_dataset = datasets.ImageFolder(root=test_data_folder_pass, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self._set_loaders(train_loader, test_loader)

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to create a SummaryWriter to
        # interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        def perform_train_step_fn(image: torch.Tensor):
            self.model.train()
            reconstructed_image = self.model(image)
            loss = self.loss_fn(reconstructed_image, image)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return perform_train_step_fn

    def _mini_batch(self):
        data_loader=self.train_loader
        step_fn=self._make_train_step_fn()
        mini_batch_losses = []
        for (image_batch, target) in data_loader:
            image_batch = image_batch.to(device=self.device)
            mini_batch_loss = step_fn(image_batch)
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        return loss

    def _print_train_loss(self, loss):
        print(f'Epoch: {self.total_epochs} | Loss: {loss.item():.4f}')

    def train(self,n_epochs):
        image, label = self.test_loader.dataset.__getitem__(0)
        image=image.unsqueeze(0)
        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss = self._mini_batch()
            self.losses.append(loss)

            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)

                reconstructed_image = self.predict(image).squeeze(0)
                self.writer.add_image(f'Images/test_image', reconstructed_image, epoch)

                for name,param in self.model.named_parameters():
                    self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            self._print_train_loss(loss)

        if self.writer is not None:
            self.writer.flush()

    def save_checkpoint(self, filename):
        checkpoint = {'epoch': self.total_epochs,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'loss': self.losses}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename,weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']

    def predict (self, image):
        self.model.eval()
        # image_tensor =  torch.as_tensor(image)
        reconstructed_image = self.model(image).to(self.device)
        self.model.train()
        return reconstructed_image

    def plot_losses(self):
        fig = plt.figure(figsize=(10,4))
        plt.plot(self.losses, label='Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return fig

    def show_reconstruction(self, image): # image - тензор [1,1,128,128]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Reconstruction')
        image2show = image.squeeze(0)
        ax1.imshow(image2show, cmap='gray')

        image2reconstruct = image
        reconstructed_image = self.predict(image2reconstruct).squeeze().detach()
        ax2.imshow(reconstructed_image, cmap='gray')
        plt.show()

    def show_error_map(self, image):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Error map')

        image2show = image.squeeze(0)
        ax1.imshow(image2show, cmap='gray')
        image2reconstruct = image.unsqueeze(0)
        reconstructed_image = self.predict(image2reconstruct).squeeze().detach()
        ax2.imshow(reconstructed_image, cmap='gray')

        error = abs(reconstructed_image - image2show)
        im = ax3.imshow(error, cmap='gray')
        plt.colorbar(im, ax=ax3)
        plt.show()


class VAEStepByStep(StepByStep):
    def _make_train_step_fn(self):
        def perform_train_step_fn(image: torch.Tensor):
            self.model.train()
            reconstructed_image, mean, log_var = self.model(image)
            loss, reconstruction_loss, kld = self.loss_fn(reconstructed_image, image, mean, log_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item(), reconstruction_loss.item(), kld.item()
        return perform_train_step_fn

    def _mini_batch(self):
        data_loader=self.train_loader
        step_fn=self._make_train_step_fn()
        mini_batch_losses = []
        mini_batch_reconstruction_losses = []
        mini_batch_klds = []
        for (image_batch, target) in data_loader:
            image_batch = image_batch.to(device=self.device)
            mini_batch_loss, reconstruction_loss, kld = step_fn(image_batch)
            mini_batch_losses.append(mini_batch_loss)
            mini_batch_reconstruction_losses.append(reconstruction_loss)
            mini_batch_klds.append(kld)
        loss = np.mean(mini_batch_losses)
        reconstruction_loss = np.mean(mini_batch_reconstruction_losses)
        kld = np.mean(mini_batch_klds)
        return loss, reconstruction_loss, kld

    def train(self,n_epochs):
        image, label = self.test_loader.dataset.__getitem__(0)
        image=image.unsqueeze(0)
        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss, reconstruction_loss, kld = self._mini_batch()
            self.losses.append(loss)

            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), epoch)
                self.writer.add_scalar('Reconstruction Loss/train', reconstruction_loss.item(), epoch)
                self.writer.add_scalar('KL Divergence/train', kld.item(), epoch)

                reconstructed_image = self.predict(image).squeeze(0)
                self.writer.add_image(f'Images/test_image', reconstructed_image, epoch)

                for name,param in self.model.named_parameters():
                    self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            self._print_train_loss(loss, reconstruction_loss, kld)

        if self.writer is not None:
            self.writer.flush()

    def set_tensorboard(self, name, folder='VAE_runs'):
        # This method allows the user to create a SummaryWriter to
        # interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def predict (self, image):
        self.model.eval()
        # image_tensor =  torch.as_tensor(image)
        # reconstructed_image, _, _ = self.model(image).to(self.device)
        reconstructed_image, _, _ = self.model(image)
        self.model.train()
        return reconstructed_image

    def _print_train_loss(self, loss, reconstruction_loss, kld):
        print(f'Epoch: {self.total_epochs} | Loss: {loss.item():.4f} | '
              f'Reconstruction Loss: {reconstruction_loss:.4f} | '
              f'KL Divergence: {kld:.8f}')