"""CPU inference + benchmark of every exported ONNX model with ONNX Runtime (CPUExecutionProvider)."""
import glob, time, csv, numpy as np, onnxruntime as ort
from common import *

_, test_dl, _ = get_loaders(128, 2)
X = np.concatenate([x.numpy() for x, _ in test_dl]); Y = np.concatenate([y.numpy() for _, y in test_dl])
rows = []
for f in sorted(glob.glob(os.path.join(ONNX, "*.onnx"))):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    pred = np.concatenate([s.run(None, {"input": X[i:i+128]})[0].argmax(1) for i in range(0, len(X), 128)])
    acc = 100 * (pred == Y).mean()
    for i in range(20): s.run(None, {"input": X[i:i+1]})
    t = []
    for i in range(200):
        st = time.perf_counter(); s.run(None, {"input": X[i:i+1]}); t.append((time.perf_counter() - st) * 1000)
    r = dict(onnx_model=os.path.basename(f), accuracy_pct=round(acc, 2), latency_mean_ms=round(np.mean(t), 3),
             latency_p99_ms=round(np.percentile(t, 99), 3), size_mb=round(file_mb(f), 3))
    rows.append(r); print(r)
with open(os.path.join(RES, "onnx_benchmark.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
