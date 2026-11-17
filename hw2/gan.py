import io
import os
import zipfile
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def load_data(zip_path="data/wikiart.zip", img_size=64):
    # decode + resize all the jpegs once and keep them as uint8
    os.makedirs("cache", exist_ok=True)
    cache_file = f"cache/wikiart_{img_size}.pt"
    if os.path.exists(cache_file):
        return torch.load(cache_file)

    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.PILToTensor(),
    ])
    # read straight from the kaggle zip, extracting it would take another 31GB
    # and some file names have broken unicode that windows doesn't like
    zf = zipfile.ZipFile(zip_path)
    files = sorted(f for f in zf.namelist() if f.lower().endswith(".jpg"))
    imgs = []
    for i, f in enumerate(files):
        try:
            img = Image.open(io.BytesIO(zf.read(f)))
            # draft lets the jpeg decoder downscale while decoding, much faster for big paintings
            img.draft("RGB", (2 * img_size, 2 * img_size))
            imgs.append(tf(img.convert("RGB")))
        except OSError:
            print("skipping broken file", f)
        if i % 5000 == 0:
            print(f"{i}/{len(files)}")
    imgs = torch.stack(imgs)
    torch.save(imgs, cache_file)
    return imgs


# DCGAN (Radford et al. 2016) for 64x64 images
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            # z: nz x 1 x 1 -> 4x4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 4 -> 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 32 -> 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), self.nz, 1, 1))

    def sample(self, n, device):
        with torch.no_grad():
            z = torch.randn(n, self.nz, device=device)
            return (self(z) + 1) / 2  # [-1,1] -> [0,1]


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # 64 -> 32, no batchnorm on the first layer as in the paper
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 32 -> 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # 16 -> 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # 8 -> 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # 4 -> 1 logit
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    # N(0, 0.02) init from the DCGAN paper
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
