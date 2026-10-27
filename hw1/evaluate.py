# FID and IS with pytorch-ignite, based on
# https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
import json
import argparse
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from torchvision.utils import save_image

from vae import VAE, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results")
args = parser.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_images = 2000
batch_size = 50

model = VAE(128).to(device)
model.load_state_dict(torch.load(f"{args.out}/vae.pt", map_location=device))
model.eval()

train_x = load_data("data/chest_xray", "train")
test_x = load_data("data/chest_xray", "test")
real_x = train_x[torch.randperm(len(train_x))[:n_images]]

# torchvision's inception_v3 expects imagenet normalized inputs
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def prep(x):
    # grayscale -> 3 channels, resize to 299x299 like in the blog
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return (x - mean) / std


def get_fid_is(fake, real):
    # fake and real are tensors with the same number of images
    def step(engine, batch):
        fake_batch, real_batch = batch
        return prep(fake_batch.to(device)), prep(real_batch.to(device))

    evaluator = Engine(step)
    FID(device=device).attach(evaluator, "fid")
    InceptionScore(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")

    data = list(zip(torch.split(fake, batch_size), torch.split(real, batch_size)))
    evaluator.run(data, max_epochs=1)
    return evaluator.state.metrics["fid"], evaluator.state.metrics["is"]


# generate samples and reconstructions
samples, recons = [], []
with torch.no_grad():
    for x in torch.split(real_x, batch_size):
        samples.append(model.sample(x.size(0), device).cpu())
        recons.append(model(x.to(device))[0].cpu())
samples = torch.cat(samples)
recons = torch.cat(recons)

results = {}
results["samples"] = get_fid_is(samples, real_x)
results["recon"] = get_fid_is(recons, real_x)
# real test images vs real train images, as a reference
results["real_test"] = get_fid_is(test_x[:600], real_x[:600])

for name, (fid, inc) in results.items():
    print(f"{name}: FID = {fid:.4f}, IS = {inc:.3f}")

with open(f"{args.out}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# wide grid of samples for the report
torch.manual_seed(1)
save_image(model.sample(48, device), f"{args.out}/samples_grid.png", nrow=16, padding=1)
