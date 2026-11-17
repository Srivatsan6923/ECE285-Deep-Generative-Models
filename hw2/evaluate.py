# FID and IS with pytorch-ignite, based on
# https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
import glob
import json
import argparse
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torchvision.utils import save_image

from gan import Generator, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results")
parser.add_argument("--n_images", type=int, default=5000)
args = parser.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 100

data = load_data()
perm = torch.randperm(len(data))
real_a = data[perm[:args.n_images]]
real_b = data[perm[args.n_images:2 * args.n_images]]

# torchvision's inception_v3 expects imagenet normalized inputs
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def prep(x):
    # x is uint8 or float in [0,1], resize to 299x299 like in the blog
    if x.dtype == torch.uint8:
        x = x.float() / 255
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return (x - mean) / std


def get_fid_is(fake, real):
    def step(engine, batch):
        fake_batch, real_batch = batch
        return prep(fake_batch.to(device)), prep(real_batch.to(device))

    evaluator = Engine(step)
    # FID() with no arguments (what the blog uses) is computed on the 1000-d softmax output,
    # not on the usual 2048-d pool features, so I also compute the standard pool-feature FID
    FID(device=device).attach(evaluator, "fid")
    pool = InceptionModel(return_features=True, device=device)
    FID(num_features=2048, feature_extractor=pool, device=device).attach(evaluator, "fid_pool")
    InceptionScore(device=device, output_transform=lambda x: x[0]).attach(evaluator, "is")

    batches = list(zip(torch.split(fake, batch_size), torch.split(real, batch_size)))
    evaluator.run(batches, max_epochs=1)
    return {k: evaluator.state.metrics[k] for k in ["fid", "fid_pool", "is"]}


def show(name, r):
    print(f"{name}: FID = {r['fid']:.4f}, FID (pool) = {r['fid_pool']:.2f}, IS = {r['is']:.3f}")


def generate(G, n):
    out = []
    for _ in range(0, n, batch_size):
        out.append(G.sample(batch_size, device).cpu())
    return torch.cat(out)[:n]


G = Generator().to(device)
results = {}

# score every saved checkpoint to see how FID moves during training
for ckpt in sorted(glob.glob(f"{args.out}/G_ep*.pt")):
    epoch = int(ckpt[-6:-3])
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()
    results[f"epoch_{epoch}"] = get_fid_is(generate(G, args.n_images), real_a)
    show(f"epoch {epoch}", results[f"epoch_{epoch}"])

# reference points: two disjoint real sets (best case up to sampling noise) and uniform noise
results["real"] = get_fid_is(real_b, real_a)
results["noise"] = get_fid_is(torch.rand(args.n_images, 3, 64, 64), real_a)
show("real", results["real"])
show("noise", results["noise"])

with open(f"{args.out}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# wide grid from the final model for the report
G.load_state_dict(torch.load(f"{args.out}/G.pt", map_location=device))
G.eval()
torch.manual_seed(1)
save_image(G.sample(48, device), f"{args.out}/samples_grid.png", nrow=16, padding=1)
save_image(real_a[:48].float() / 255, f"{args.out}/real_grid.png", nrow=16, padding=1)
