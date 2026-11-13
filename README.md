# Edge Computing & Intelligent Systems - Assignment 03
Roll no: 22i-2063 | Model: CIFAR-style ResNet-18 on CIFAR-10 | PyTorch (eager-mode quantization + torch.nn.utils.prune) + ONNX Runtime

## 1. Installation
```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```
Python 3.9+. A GPU (e.g. Colab T4) is only needed to speed up training/QAT/fine-tuning; quantized inference and benchmarking run on CPU. CIFAR-10 downloads automatically into `./data`.

## 2. Step-by-step execution (or just `bash run_all.sh`)
| Step | Command | Produces |
|---|---|---|
| 1 | `python train_baseline.py --epochs 30` | `checkpoints/resnet18_fp32.pth` |
| 2 | `python ptq.py` | `checkpoints/resnet18_ptq_int8.pt` |
| 3 | `python qat.py --epochs 6` | `checkpoints/resnet18_qat_int8.pt` |
| 4 | `python prune_unstructured.py --target-sparsity 0.8 --iterations 5` | `checkpoints/resnet18_unstructured_pruned.pth` |
| 5 | `python prune_structured.py --ratio 0.5 --ft-epochs 10` | `checkpoints/resnet18_structured_pruned.pth` |
| 6 | `python benchmark.py` | `results/benchmark_results.csv`, `results/predictions.csv` |
| 7 | `python export_onnx.py` | `onnx_models/*.onnx` (fp32, unstructured, structured, INT8) |
| 8 | `python onnx_inference.py` | `results/onnx_benchmark.csv` |

Single-model inference: `python inference.py --model {fp32|ptq_int8|qat_int8|unstructured|structured} --image img.png` or `--test-samples 10`.

## 3. File map
- `common.py` - data, model, evaluation, checkpoint loaders. `train_baseline.py` - FP32 training.
- `ptq.py` (fuse, backend qconfig, calibration on 2000 train images, convert) / `qat.py` (init from FP32, fake-quant, fine-tune, convert).
- `prune_unstructured.py` (iterative global L1 magnitude pruning, masks, fine-tune, `prune.remove`) / `prune_structured.py` (channel pruning + rebuilt smaller architecture + fine-tune).
- `inference.py`, `benchmark.py`, `export_onnx.py`, `onnx_inference.py`, `REPORT.docx`.

## 4. Notes
- Seed fixed to 42. INT8 models are saved as TorchScript so they load without the class definition (`torch.jit.load`).
- INT8 PyTorch models run on CPU only (fbgemm on x86 / qnnpack on ARM, selected automatically).
- Structured pruning only prunes the *inner* channels of each residual block, so skip connections stay dimensionally valid.
- Smoke test without downloading data: `CIFAR_FAKE=1 python train_baseline.py --epochs 1` (synthetic data, numbers meaningless).
- Large `.pth/.pt/.onnx` files are the required trained models (deliverables), not executables.
