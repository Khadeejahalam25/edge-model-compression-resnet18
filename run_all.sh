#!/usr/bin/env bash
# Reproducible end-to-end pipeline (seed fixed to 42). On Colab: Runtime > T4 GPU, then `bash run_all.sh`
set -e
pip install -q -r requirements.txt
python train_baseline.py --epochs 30
python ptq.py
python qat.py --epochs 6
python prune_unstructured.py --target-sparsity 0.8 --iterations 5
python prune_structured.py --ratio 0.5 --ft-epochs 10
python benchmark.py
python export_onnx.py
python onnx_inference.py
