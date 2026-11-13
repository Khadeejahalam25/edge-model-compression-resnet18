"""Q-ONNX - Export deployable ONNX models.
FP32 / unstructured / structured -> ONNX directly. INT8 ONNX is produced with ONNX Runtime static quantization
(QDQ format, calibrated on training images), because PyTorch eager-mode INT8 ops are not exportable to ONNX."""
import numpy as np, torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from common import *

def export(name):
    m = load_model(name).eval()
    out = os.path.join(ONNX, f"resnet18_{name}.onnx")
    dummy = torch.randn(1, 3, 32, 32)
    kw = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(m, dummy, out, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, dummy, out, **kw)
    print("exported", out); return out

class Reader(CalibrationDataReader):
    def __init__(self, n=256):
        _, _, tr = get_loaders(32, 0)
        idx = torch.randperm(len(tr))[:n].tolist()
        self.data = iter([{"input": tr[i][0].unsqueeze(0).numpy().astype(np.float32)} for i in idx])
    def get_next(self):
        return next(self.data, None)

seed_everything()
for n in ("fp32", "unstructured", "structured"):
    export(n)
fp32 = os.path.join(ONNX, "resnet18_fp32.onnx")
pre = os.path.join(ONNX, "_fp32_pre.onnx")
try:
    quant_pre_process(fp32, pre); src = pre
except Exception:
    src = fp32
out = os.path.join(ONNX, "resnet18_ptq_int8.onnx")
quantize_static(src, out, Reader(), quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
print("exported INT8 ONNX", out)
if os.path.exists(pre): os.remove(pre)
