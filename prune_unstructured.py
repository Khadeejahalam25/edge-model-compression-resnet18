"""Q2.2 + Q2.4 - Iterative global magnitude (L1) unstructured pruning with masking, fine-tuning, and permanent removal."""
import argparse, torch, torch.nn as nn
import torch.nn.utils.prune as prune
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--target-sparsity", type=float, default=0.8)
ap.add_argument("--iterations", type=int, default=5)
ap.add_argument("--ft-epochs", type=int, default=2, help="fine-tune epochs after each pruning iteration")
ap.add_argument("--final-epochs", type=int, default=5, help="extra fine-tune epochs after the last iteration")
ap.add_argument("--lr", type=float, default=0.01)
a = ap.parse_args()
seed_everything(); dev = get_device()
train_dl, test_dl, _ = get_loaders(128)

model = build_model(); model.load_state_dict(torch.load(PATHS["fp32"], map_location="cpu")); model.to(dev)
params = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
print(f"Baseline acc: {evaluate(model, test_dl, dev):.2f}%")

# amount pruned each iteration (of the REMAINING weights) so that final sparsity == target
per_iter = 1 - (1 - a.target_sparsity) ** (1 / a.iterations)

def finetune(epochs, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        train_one_epoch(model, train_dl, opt, dev)

def sparsity():
    tot = sum(m.weight.nelement() for m, _ in params)
    z = sum((m.weight == 0).sum().item() for m, _ in params)
    return 100.0 * z / tot

for it in range(1, a.iterations + 1):
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=per_iter)  # masks (weight_orig * mask)
    before = evaluate(model, test_dl, dev)
    finetune(a.ft_epochs, a.lr)                      # masks stay active during fine-tuning
    print(f"iter {it}/{a.iterations}: sparsity {sparsity():.1f}%  acc after prune {before:.2f}% -> after FT {evaluate(model, test_dl, dev):.2f}%")

finetune(a.final_epochs, a.lr / 5)
for m, n in params:
    prune.remove(m, n)                               # make pruning permanent (bake mask into weight)
acc = evaluate(model, test_dl, dev)
print(f"FINAL: weight sparsity {sparsity():.2f}%  test acc {acc:.2f}%")
torch.save(model.cpu().state_dict(), PATHS["unstructured"])
