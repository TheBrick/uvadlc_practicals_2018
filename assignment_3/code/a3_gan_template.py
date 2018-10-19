import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.generator = nn.Sequential(
            nn.Linear(args.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        img = self.generator(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.discriminator = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        probability_of_real_img = self.discriminator(img)
        return probability_of_real_img


def save_GAN_plot(G_curve, D_curve, filename):
    plt.figure(figsize=(12, 6))
    line1 = plt.plot(G_curve, label='G loss')
    line2 = plt.plot(D_curve, label='D loss')
    plt.setp(line1, linewidth=0.25)
    plt.setp(line2, linewidth=0.25)
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig(filename)

def train(dataloader, D, G, optimizer_G, optimizer_D):
    D_curve, G_curve = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Prepare real images
            X = imgs.view(-1, 784).to(device)

            # Train Generator
            # ---------------
            Z = torch.randn(X.shape[0], args.latent_dim, device=device)
            loss_G = -torch.log(D(G(Z))).sum()
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            Z = torch.randn(X.shape[0], args.latent_dim, device=device)  # resample Z
            loss_D = -(torch.log(D(X)) + torch.log(1 - D(G(Z)))).sum()
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Save them for later
            G_curve.append(loss_G)
            D_curve.append(loss_D)

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(G(Z[:25]).view(-1, 1, 28, 28),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
                print("Epoch:{} Iteration:{} loss_D:{:.2f} loss_G:{:.2f}".format(epoch, i, loss_D, loss_G))

    save_GAN_plot(G_curve, D_curve, 'GAN.pdf')


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator_epoch={}.pt".format(args.n_epochs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
