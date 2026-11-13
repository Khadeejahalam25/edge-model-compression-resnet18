"""Q1.3 - Quantization-Aware Training: init from FP32 -> fuse -> fake-quant -> fine-tune -> convert -> export."""
import argparse, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()
seed_everything()

backend = get_backend()
torch.backends.quantized.engine = backend
dev = get_device()
train_dl, test_dl, _ = get_loaders(128)

model = build_model(); model.load_state_dict(torch.load(PATHS["fp32"], map_location="cpu"))  # pretrained FP32 init
model.train()
fuse(model, is_qat=True)
model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
torch.ao.quantization.prepare_qat(model, inplace=True)       # inserts FakeQuantize modules
model.to(dev)

opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(train_dl))
for ep in range(1, a.epochs + 1):
    if ep == max(2, a.epochs // 2):   # freeze observers / BN stats in the second half
        model.apply(torch.ao.quantization.disable_observer)
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    loss = train_one_epoch(model, train_dl, opt, dev, sched)
    print(f"QAT epoch {ep}/{a.epochs}  loss {loss:.4f}  fake-quant test acc {evaluate(model, test_dl, dev):.2f}%")

model.cpu().eval()
int8 = torch.ao.quantization.convert(model, inplace=False)   # fake-quant -> real INT8
torch.save(int8.state_dict(), os.path.join(CKPT, "resnet18_qat_int8_state_dict.pth"))
save_int8(int8, PATHS["qat_int8"], torch.randn(1, 3, 32, 32))
print(f"QAT INT8 test accuracy: {evaluate(load_model('qat_int8'), test_dl):.2f}%  | size {file_mb(PATHS['qat_int8']):.2f} MB")
