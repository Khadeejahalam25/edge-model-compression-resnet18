"""Q2.3 + Q2.4 - Structured channel pruning of ResNet-18 + architecture reconstruction + fine-tuning.

Strategy: inside every BasicBlock, rank the filters of conv1 by L1 norm and keep the top (1-ratio).
conv1's output channels, bn1 and conv2's INPUT channels are physically removed. The block output channels
(conv2 out / residual path) are untouched, so every skip-connection addition stays dimensionally valid.
The result is a genuinely smaller dense network (no masked zero channels)."""
import argparse, torch, torch.nn as nn
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ratio", type=float, default=0.5, help="fraction of inner channels removed per block")
ap.add_argument("--ft-epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=0.01)
a = ap.parse_args()
seed_everything(); dev = get_device()
train_dl, test_dl, _ = get_loaders(128)

model = build_model(); model.load_state_dict(torch.load(PATHS["fp32"], map_location="cpu"))
print(f"Baseline acc: {evaluate(model, test_dl, 'cpu') if dev.type == 'cpu' else evaluate(model, test_dl, dev):.2f}%")
model.cpu()

blocks = [b for l in (model.layer1, model.layer2, model.layer3, model.layer4) for b in l]
cfg, keep_idx = [], []
for b in blocks:
    imp = b.conv1.weight.detach().abs().sum(dim=(1, 2, 3))            # L1 importance per output filter
    k = max(8, int(round(imp.numel() * (1 - a.ratio))))
    keep = torch.sort(torch.topk(imp, k).indices).values
    cfg.append(k); keep_idx.append(keep)

# Reconstruct a smaller architecture and copy the surviving weights
new = build_structured(cfg)
sd = model.state_dict(); new_sd = new.state_dict()
for k_, v in sd.items():                       # copy everything with identical shape first
    if k_ in new_sd and new_sd[k_].shape == v.shape:
        new_sd[k_] = v.clone()
names = [f"layer{i}.{j}" for i, l in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1) for j in range(len(l))]
for name, keep in zip(names, keep_idx):
    new_sd[f"{name}.conv1.weight"] = sd[f"{name}.conv1.weight"][keep].clone()
    for s in ("weight", "bias", "running_mean", "running_var"):
        new_sd[f"{name}.bn1.{s}"] = sd[f"{name}.bn1.{s}"][keep].clone()
    new_sd[f"{name}.conv2.weight"] = sd[f"{name}.conv2.weight"][:, keep].clone()
new.load_state_dict(new_sd)
n_old = sum(p.numel() for p in model.parameters()); n_new = sum(p.numel() for p in new.parameters())
print(f"Kept channels per block: {cfg}\nParams {n_old/1e6:.2f}M -> {n_new/1e6:.2f}M ({100*(1-n_new/n_old):.1f}% fewer)")
new.to(dev)
print(f"Accuracy right after pruning (before FT): {evaluate(new, test_dl, dev):.2f}%")

opt = torch.optim.SGD(new.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.ft_epochs * len(train_dl))
for ep in range(1, a.ft_epochs + 1):
    loss = train_one_epoch(new, train_dl, opt, dev, sched)
    print(f"FT epoch {ep}/{a.ft_epochs}  loss {loss:.4f}  test acc {evaluate(new, test_dl, dev):.2f}%")
torch.save({"cfg": cfg, "state_dict": new.cpu().state_dict()}, PATHS["structured"])
print("Saved", PATHS["structured"])
