import os
import json
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from vae import VAE, vae_loss, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/chest_xray")
parser.add_argument("--latent", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--out", default="results")
args = parser.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)

train_x = load_data(args.data, "train")
test_x = load_data(args.data, "test")
print("train:", train_x.shape, "test:", test_x.shape)

train_loader = DataLoader(TensorDataset(train_x), batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(test_x), batch_size=args.batch_size)

model = VAE(args.latent).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
print("num params:", sum(p.numel() for p in model.parameters()))

# same z every time so i can see how the samples change during training
fixed_z = torch.randn(64, args.latent, device=device)

train_loss, train_bce, train_kl = [], [], []
test_loss, test_bce, test_kl = [], [], []

for epoch in range(1, args.epochs + 1):
    model.train()
    total, total_bce, total_kl = 0, 0, 0
    for (x,) in train_loader:
        x = x.to(device)
        x_recon, mu, logvar = model(x)
        loss, bce, kl = vae_loss(x_recon, x, mu, logvar, args.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        total_bce += bce.item()
        total_kl += kl.item()
    scheduler.step()
    train_loss.append(total / len(train_loader))
    train_bce.append(total_bce / len(train_loader))
    train_kl.append(total_kl / len(train_loader))

    model.eval()
    total, total_bce, total_kl = 0, 0, 0
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, bce, kl = vae_loss(x_recon, x, mu, logvar, args.beta)
            total += loss.item() * x.size(0)
            total_bce += bce.item() * x.size(0)
            total_kl += kl.item() * x.size(0)
    test_loss.append(total / len(test_x))
    test_bce.append(total_bce / len(test_x))
    test_kl.append(total_kl / len(test_x))

    print(f"epoch {epoch} train loss {train_loss[-1]:.1f} (bce {train_bce[-1]:.1f}, kl {train_kl[-1]:.1f}) test loss {test_loss[-1]:.1f}")

    if epoch == 1 or epoch % 10 == 0:
        with torch.no_grad():
            save_image(model.decode(fixed_z), f"{args.out}/samples_ep{epoch:03d}.png", nrow=8)

torch.save(model.state_dict(), f"{args.out}/vae.pt")

history = {
    "train_loss": train_loss, "train_bce": train_bce, "train_kl": train_kl,
    "test_loss": test_loss, "test_bce": test_bce, "test_kl": test_kl,
}
with open(f"{args.out}/history.json", "w") as f:
    json.dump(history, f)

# loss curves
epochs = range(1, args.epochs + 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
axes[0].plot(epochs, train_loss, label="train")
axes[0].plot(epochs, test_loss, label="test")
axes[0].set_title("Total loss (-ELBO)")
axes[1].plot(epochs, train_bce, label="train")
axes[1].plot(epochs, test_bce, label="test")
axes[1].set_title("Reconstruction (BCE)")
axes[2].plot(epochs, train_kl, label="train")
axes[2].plot(epochs, test_kl, label="test")
axes[2].set_title("KL divergence")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("nats per image")
axes[0].legend()
plt.tight_layout()
plt.savefig(f"{args.out}/loss_curves.pdf")
plt.savefig(f"{args.out}/loss_curves.png", dpi=150)

# final samples and some reconstructions from the test set
model.eval()
save_image(model.sample(64, device), f"{args.out}/samples_final.png", nrow=8)
with torch.no_grad():
    x = test_x[torch.randperm(len(test_x))[:8]].to(device)
    x_recon, _, _ = model(x)
    save_image(torch.cat([x, x_recon]), f"{args.out}/recon.png", nrow=8)
