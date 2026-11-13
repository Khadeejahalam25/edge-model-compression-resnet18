"""Q1.1 / Q2.1 - Train the FP32 ResNet-18 baseline on CIFAR-10 and save the checkpoint."""
import argparse, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--bs", type=int, default=128)
a = ap.parse_args()

seed_everything(); dev = get_device()
train_dl, test_dl, _ = get_loaders(a.bs)
model = build_model().to(dev)
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(train_dl))
scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

best = 0.0
for ep in range(1, a.epochs + 1):
    loss = train_one_epoch(model, train_dl, opt, dev, sched, scaler)
    acc = evaluate(model, test_dl, dev)
    print(f"epoch {ep:02d}/{a.epochs}  loss {loss:.4f}  test acc {acc:.2f}%")
    if acc >= best:
        best = acc
        torch.save(model.state_dict(), PATHS["fp32"])
print(f"Best FP32 test accuracy: {best:.2f}%  -> saved {PATHS['fp32']}")
