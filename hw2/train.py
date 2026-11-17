import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from gan import Generator, Discriminator, init_weights, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/wikiart.zip")
parser.add_argument("--nz", type=int, default=100)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--g_loss", default="ns", choices=["ns", "minimax"])
parser.add_argument("--real_label", type=float, default=1.0)  # 0.9 = one-sided label smoothing
parser.add_argument("--out", default="results")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)

# the whole dataset is ~1GB as uint8, so it fits on the gpu and we skip the DataLoader
data = load_data(args.data).to(device)
n = len(data)
print("images:", data.shape)

G = Generator(args.nz).to(device)
D = Discriminator().to(device)
G.apply(init_weights)
D.apply(init_weights)
opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
print("G params:", sum(p.numel() for p in G.parameters()))
print("D params:", sum(p.numel() for p in D.parameters()))

fixed_z = torch.randn(64, args.nz, device=device)
ones = torch.ones(args.batch_size, device=device)
zeros = torch.zeros(args.batch_size, device=device)
real_target = torch.full((args.batch_size,), args.real_label, device=device)

# per iteration logs
hist = {"loss_d": [], "loss_g": [], "d_real": [], "d_fake": []}
iters_per_epoch = n // args.batch_size
start = time.time()

for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(n, device=device)
    for it in range(iters_per_epoch):
        x = data[perm[it * args.batch_size:(it + 1) * args.batch_size]]
        x = x.float() / 127.5 - 1  # uint8 -> [-1, 1]
        flip = torch.rand(x.size(0), device=device) < 0.5
        x[flip] = x[flip].flip(3)

        # D step: maximize log D(x) + log(1 - D(G(z)))
        z = torch.randn(args.batch_size, args.nz, device=device)
        fake = G(z)
        out_real = D(x)
        out_fake = D(fake.detach())
        loss_d = F.binary_cross_entropy_with_logits(out_real, real_target) + \
                 F.binary_cross_entropy_with_logits(out_fake, zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        out_fake = D(fake)
        if args.g_loss == "ns":
            # non-saturating: maximize log D(G(z))
            loss_g = F.binary_cross_entropy_with_logits(out_fake, ones)
        else:
            # original minimax: minimize log(1 - D(G(z))), gradient vanishes when D wins
            loss_g = -F.binary_cross_entropy_with_logits(out_fake, zeros)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        hist["loss_d"].append(loss_d.item())
        hist["loss_g"].append(loss_g.item())
        hist["d_real"].append(torch.sigmoid(out_real).mean().item())
        hist["d_fake"].append(torch.sigmoid(out_fake).mean().item())

    e = slice(-iters_per_epoch, None)
    print(f"epoch {epoch} loss_d {np.mean(hist['loss_d'][e]):.3f} loss_g {np.mean(hist['loss_g'][e]):.3f} "
          f"D(x) {np.mean(hist['d_real'][e]):.3f} D(G(z)) {np.mean(hist['d_fake'][e]):.3f} "
          f"({time.time() - start:.0f}s)")

    if epoch == 1 or epoch % 5 == 0:
        G.eval()
        with torch.no_grad():
            save_image((G(fixed_z) + 1) / 2, f"{args.out}/samples_ep{epoch:03d}.png", nrow=8)
        G.train()
        torch.save(G.state_dict(), f"{args.out}/G_ep{epoch:03d}.pt")
        # save the logs with every checkpoint so an interrupted run keeps them
        with open(f"{args.out}/history.json", "w") as f:
            json.dump(hist, f)

torch.save(G.state_dict(), f"{args.out}/G.pt")
torch.save(D.state_dict(), f"{args.out}/D.pt")
with open(f"{args.out}/history.json", "w") as f:
    json.dump(hist, f)


# loss curves, raw per iteration in light color and a moving average on top
def smooth(y, k=200):
    k = min(k, len(y))
    return np.convolve(y, np.ones(k) / k, mode="valid")


ep = np.arange(len(hist["loss_d"])) / iters_per_epoch
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for key, label, c in [("loss_d", "D loss", "C0"), ("loss_g", "G loss", "C1")]:
    axes[0].plot(ep, hist[key], color=c, alpha=0.15, lw=0.5)
    axes[0].plot(ep[len(ep) - len(smooth(hist[key])):], smooth(hist[key]), color=c, label=label)
axes[0].set_title("Losses")
axes[0].set_ylabel("loss (nats)")
for key, label, c in [("d_real", "D(x)", "C2"), ("d_fake", "D(G(z))", "C3")]:
    axes[1].plot(ep, hist[key], color=c, alpha=0.15, lw=0.5)
    axes[1].plot(ep[len(ep) - len(smooth(hist[key])):], smooth(hist[key]), color=c, label=label)
axes[1].set_title("Discriminator outputs")
axes[1].set_ylim(0, 1)
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig(f"{args.out}/loss_curves.pdf")
plt.savefig(f"{args.out}/loss_curves.png", dpi=150)

G.eval()
save_image(G.sample(64, device), f"{args.out}/samples_final.png", nrow=8)
