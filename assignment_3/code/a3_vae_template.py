import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.encoder_mean = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        self.encoder_std = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean = self.encoder_mean(input)
        std = self.encoder_std(input)
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.decoder(input)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, device='cpu'):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim).to(device)
        self.decoder = Decoder(hidden_dim, z_dim).to(device)
        self.device = device

    def log_bernoulli_loss (self, x, r):
        e = 1e-8  # nonzerofier
        losses = - torch.sum(x * torch.log(r+e) + (1 - x) * torch.log(1 - r+e), dim=1)
        return losses

    def KL_loss (self, mu, sigma):
        # sum over components and data points at the same time (summing loss of each data point)
        wip = 1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        divergence = torch.sum(wip, dim=1) / 2
        return -divergence

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # Bark bark
        input = input.view(-1, 784).to(self.device)

        # Encode input onto Z
        mean, std = self.encoder(input)

        # Sample Z with reparametrization trick
        epsilons = torch.randn((1, self.z_dim), device=self.device)
        z = mean + std * epsilons

        # Reconstruct inputs with decoder
        reconstructions = self.decoder(z)

        # Calculate the reconstruction loss
        reconstruction_losses = self.log_bernoulli_loss(input, reconstructions)

        # Calculate the regularization loss
        regularization_losses = self.KL_loss(mean, std)

        # Calculate ELBO
        average_negative_elbo = torch.mean(reconstruction_losses + regularization_losses, dim=0)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        # Sample Z without transforming prior
        z = torch.randn((n_samples, self.z_dim), device=self.device)

        # Construct samples with decoder
        sampled_ims = self.decoder(z)

        # Calculate means
        im_means = sampled_ims.mean(dim=0)

        # Convert to binary to get actual samples, but let's not do it here to
        # get more insight into the decoder from the images
        # sampled_ims = (samples > 0.5) * 255


        return sampled_ims.view(n_samples, 1, 28, 28), im_means.view(1, 1, 28, 28)


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    total_elbo = 0

    for i, batch in enumerate(data):
        avg_elbo = model(batch)
        if model.training:
            model.zero_grad()
            avg_elbo.backward()
            optimizer.step()
        total_elbo += avg_elbo.item()

    avg_epoch_elbo = total_elbo / len(data)
    return avg_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def sample_model (model, epoch, train_elbo, val_elbo):
    with torch.no_grad():
        row_count = 5
        samples, im_means = model.sample(row_count * row_count)
        samples = make_grid(samples, nrow=row_count)
        path = "samples/z={}_epoch={}_trainingloss={:.3f}_validationloss={:.2f}.png".format(
            ARGS.zdim,
            epoch,
            train_elbo,
            val_elbo
        )
        plt.imsave(path, samples.cpu().numpy().transpose(1, 2, 0))

def plot_manifold (model):
    with torch.no_grad():
        row_count = 20
        grid = torch.linspace(0, 1, row_count)
        samples = [torch.erfinv(2 * torch.tensor([x, y]) - 1) * math.sqrt(2) for x in grid for y in grid]
        samples = torch.stack(samples).to(model.device)
        manifold = model.decoder(samples).view(-1, 1, 28, 28)
        img = make_grid(manifold, nrow=row_count)
        plt.imsave("manifold.png", img.cpu().numpy().transpose(1, 2, 0))

def main():
    data = bmnist()[:2]  # ignore test split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE(z_dim=ARGS.zdim, device=device)
    optimizer = torch.optim.Adam(model.parameters())

    # sample before training
    sample_model(model, 0, 9999, 9999)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        sample_model(model, epoch, train_elbo, val_elbo)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        plot_manifold(model)

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
