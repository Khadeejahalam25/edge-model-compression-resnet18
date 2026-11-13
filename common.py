"""Shared utilities: data, model definition, evaluation, checkpoint loading."""
import os, time, random
import numpy as np
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from torchvision.models.quantization import resnet18 as q_resnet18

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "checkpoints")
RES = os.path.join(ROOT, "results")
ONNX = os.path.join(ROOT, "onnx_models")
DATA = os.path.join(ROOT, "data")
for d in (CKPT, RES, ONNX):
    os.makedirs(d, exist_ok=True)

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

PATHS = {  # model name -> checkpoint file
    "fp32": os.path.join(CKPT, "resnet18_fp32.pth"),
    "ptq_int8": os.path.join(CKPT, "resnet18_ptq_int8.pt"),
    "qat_int8": os.path.join(CKPT, "resnet18_qat_int8.pt"),
    "unstructured": os.path.join(CKPT, "resnet18_unstructured_pruned.pth"),
    "structured": os.path.join(CKPT, "resnet18_structured_pruned.pth"),
}


def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_backend():
    """Pick the quantization engine (fbgemm for x86, qnnpack for ARM)."""
    engines = torch.backends.quantized.supported_engines
    return "fbgemm" if "fbgemm" in engines else "qnnpack"


def get_loaders(batch_size=128, workers=2):
    """CIFAR-10 loaders. Set env CIFAR_FAKE=1 for a tiny synthetic dataset (smoke tests only)."""
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    if os.environ.get("CIFAR_FAKE") == "1":
        mk = lambda n, tf: torchvision.datasets.FakeData(n, (3, 32, 32), 10, transform=tf)
        train, train_plain, test = mk(512, train_tf), mk(512, test_tf), mk(256, test_tf)
    else:
        D = torchvision.datasets.CIFAR10
        train = D(DATA, train=True, download=True, transform=train_tf)
        train_plain = D(DATA, train=True, download=True, transform=test_tf)  # no augmentation (calibration)
        test = D(DATA, train=False, download=True, transform=test_tf)
    mk_dl = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=sh, num_workers=workers)
    return mk_dl(train, True), mk_dl(test, False), train_plain


def build_model(num_classes=10):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool) using the quantization-ready torchvision class
    (FloatFunctional skip-add, Quant/DeQuant stubs, fuse_model()). Identical code path for all models."""
    m = q_resnet18(weights=None, quantize=False)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes)
    return m


def fuse(model, is_qat):
    try:
        model.fuse_model(is_qat=is_qat)
    except TypeError:  # older torchvision
        model.fuse_model()


@torch.no_grad()
def evaluate(model, loader, device="cpu", return_preds=False):
    model.eval().to(device)
    correct = total = 0
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        p = out.argmax(1).cpu()
        correct += (p == y).sum().item(); total += y.numel()
        preds.append(p); labels.append(y)
    acc = 100.0 * correct / total
    if return_preds:
        return acc, torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return acc


def train_one_epoch(model, loader, opt, device, sched=None, scaler=None):
    model.train()
    crit = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            loss = crit(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item() * y.size(0); n += y.size(0)
    return tot / n


def save_int8(model, path, example):
    """Save a converted INT8 model as TorchScript (loadable without the class definition)."""
    model.eval().cpu()
    try:
        ts = torch.jit.script(model)
    except Exception:
        ts = torch.jit.trace(model, example)
    ts.save(path)


def build_structured(cfg):
    """Rebuild the physically-smaller network. cfg[i] = kept inner channels for each BasicBlock (in order)."""
    m = build_model()
    blocks = [b for l in (m.layer1, m.layer2, m.layer3, m.layer4) for b in l]
    for b, k in zip(blocks, cfg):
        b.conv1 = nn.Conv2d(b.conv1.in_channels, k, 3, b.conv1.stride, 1, bias=False)
        b.bn1 = nn.BatchNorm2d(k)
        b.conv2 = nn.Conv2d(k, b.conv2.out_channels, 3, 1, 1, bias=False)
    return m


def load_model(name):
    """Return an eval-mode model ready for inference. INT8 models are CPU-only."""
    p = PATHS[name]
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing - run the corresponding script first")
    if name in ("ptq_int8", "qat_int8"):
        return torch.jit.load(p, map_location="cpu").eval()
    if name == "structured":
        ck = torch.load(p, map_location="cpu")
        m = build_structured(ck["cfg"]); m.load_state_dict(ck["state_dict"])
        return m.eval()
    m = build_model(); m.load_state_dict(torch.load(p, map_location="cpu")); return m.eval()


def file_mb(path):
    return os.path.getsize(path) / 1e6


def sparsity_stats(model):
    tot = zeros = 0
    for p in model.parameters():
        tot += p.numel(); zeros += (p == 0).sum().item()
    return tot, zeros
