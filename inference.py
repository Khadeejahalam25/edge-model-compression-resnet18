"""Inference pipeline for all five models: fp32, ptq_int8, qat_int8, unstructured, structured.
Examples:
  python inference.py --model qat_int8 --image cat.png
  python inference.py --model structured --test-samples 10"""
import argparse, torch
from PIL import Image
import torchvision.transforms as T
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(PATHS))
ap.add_argument("--image", help="path to an image file (resized to 32x32)")
ap.add_argument("--test-samples", type=int, default=0, help="run on first N CIFAR-10 test images")
a = ap.parse_args()
if "int8" in a.model:
    torch.backends.quantized.engine = get_backend()
model = load_model(a.model)
tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, STD)])

def predict(x):
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    c = int(p.argmax()); return CLASSES[c], float(p[c])

if a.image:
    name, conf = predict(tf(Image.open(a.image).convert("RGB")).unsqueeze(0))
    print(f"[{a.model}] {a.image}: {name} ({conf:.1%})")
if a.test_samples:
    _, test_dl, _ = get_loaders(1, 0)
    ds = test_dl.dataset
    for i in range(a.test_samples):
        x, y = ds[i]; name, conf = predict(x.unsqueeze(0))
        print(f"[{a.model}] #{i}: true={CLASSES[y]:<10} pred={name:<10} conf={conf:.1%} {'OK' if name == CLASSES[y] else 'WRONG'}")
if not a.image and not a.test_samples:
    print("Nothing to do: pass --image and/or --test-samples")
