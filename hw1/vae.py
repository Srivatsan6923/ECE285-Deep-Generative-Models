import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_data(root, split, img_size=128):
    # loading the jpegs every epoch was slow, so i resize them once and save a .pt file
    os.makedirs("cache", exist_ok=True)
    cache_file = f"cache/{split}_{img_size}.pt"
    if os.path.exists(cache_file):
        return torch.load(cache_file)

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    files = sorted(glob.glob(os.path.join(root, split, "*", "*.jpeg")))
    imgs = []
    for f in files:
        imgs.append(tf(Image.open(f)))
    imgs = torch.stack(imgs)
    torch.save(imgs, cache_file)
    return imgs


def down_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    )


def up_block(c_in, c_out):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2),
    )


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # 128x128 -> 4x4
        self.encoder = nn.Sequential(
            down_block(1, 32),
            down_block(32, 64),
            down_block(64, 128),
            down_block(128, 256),
            down_block(256, 256),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # 4x4 -> 128x128
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            up_block(256, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 256, 4, 4)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar

    def sample(self, n, device):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            return self.decode(z)


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    # sum over pixels, mean over batch
    bce = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / x.size(0)
    return bce + beta * kl, bce, kl
