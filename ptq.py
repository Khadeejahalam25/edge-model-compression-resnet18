"""Q1.2 - Post-Training Static Quantization (PTQ): fuse -> qconfig -> calibrate -> convert -> save."""
import argparse, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--calib-samples", type=int, default=2000, help="size of the training subset used for calibration")
a = ap.parse_args()
seed_everything()

backend = get_backend()
torch.backends.quantized.engine = backend
print("Quantization backend:", backend)

_, test_dl, train_plain = get_loaders(128)
# calibration data: random subset of the TRAINING set (no augmentation)
idx = torch.randperm(len(train_plain))[: a.calib_samples].tolist()
calib_dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_plain, idx), batch_size=64)

model = build_model(); model.load_state_dict(torch.load(PATHS["fp32"], map_location="cpu")); model.eval()
fuse(model, is_qat=False)                                   # 1) fuse Conv+BN(+ReLU)
model.qconfig = torch.ao.quantization.get_default_qconfig(backend)  # 2) backend qconfig
torch.ao.quantization.prepare(model, inplace=True)          # 3) insert observers
with torch.no_grad():                                       # 4) calibration
    for i, (x, _) in enumerate(calib_dl):
        model(x)
print(f"Calibrated on {len(idx)} training images")
torch.ao.quantization.convert(model, inplace=True)          # 5) FP32 -> INT8

torch.save(model.state_dict(), os.path.join(CKPT, "resnet18_ptq_int8_state_dict.pth"))
save_int8(model, PATHS["ptq_int8"], torch.randn(1, 3, 32, 32))  # 6) deployable TorchScript INT8
print(f"PTQ INT8 test accuracy: {evaluate(load_model('ptq_int8'), test_dl):.2f}%  | size {file_mb(PATHS['ptq_int8']):.2f} MB")
