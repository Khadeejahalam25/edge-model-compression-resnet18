"""Q1.5 / Q2.6 - Automated benchmark of all 5 models on the CIFAR-10 test set (CPU, for a fair INT8 comparison).
Outputs: results/benchmark_results.csv (accuracy, latency, size, sparsity) and results/predictions.csv (per-image predictions)."""
import argparse, csv, time, numpy as np, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--lat-samples", type=int, default=300, help="images for batch-1 latency measurement")
ap.add_argument("--threads", type=int, default=1, help="CPU threads (1 = edge-like)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.backends.quantized.engine = get_backend()
_, test_dl, _ = get_loaders(128, 2)
ds = test_dl.dataset

def latency_ms(model, n):
    xs = [ds[i][0].unsqueeze(0) for i in range(min(n, len(ds)))]
    with torch.no_grad():
        for x in xs[:20]: model(x)                      # warm-up
        t = []
        for x in xs:
            s = time.perf_counter(); model(x); t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return t.mean(), np.percentile(t, 50), np.percentile(t, 99)

rows, preds, labels = [], {}, None
for name in PATHS:
    try:
        m = load_model(name)
    except FileNotFoundError as e:
        print("SKIP", name, "-", e); continue
    acc, p, y = evaluate(m, test_dl, "cpu", return_preds=True)
    mean, p50, p99 = latency_ms(m, a.lat_samples)
    path = PATHS[name]
    if name in ("ptq_int8", "qat_int8"):
        # parameters are packed inside TorchScript INT8 modules, so counts are not reported
        tot, zeros, params = "n/a", "n/a", "n/a"
        sp = "n/a"
    else:
        t, z = sparsity_stats(m); tot, zeros = t, z
        params = t - z; sp = round(100 * z / t, 2)
        tot = t
    rows.append(dict(model=name, test_accuracy_pct=round(acc, 2), latency_mean_ms=round(mean, 3),
                     latency_p50_ms=round(p50, 3), latency_p99_ms=round(p99, 3),
                     throughput_img_s=round(1000 / mean, 1), size_mb=round(file_mb(path), 3),
                     total_params=tot, nonzero_params=params, sparsity_pct=sp, threads=a.threads))
    preds[name] = p; labels = y
    print(rows[-1])

with open(os.path.join(RES, "benchmark_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(RES, "predictions.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["index", "true_label", "true_class"] + [f"pred_{n}" for n in preds])
    for i in range(len(labels)):
        w.writerow([i, labels[i], CLASSES[labels[i]]] + [int(preds[n][i]) for n in preds])
print("Wrote results/benchmark_results.csv and results/predictions.csv")
