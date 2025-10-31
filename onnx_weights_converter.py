from pathlib import Path
from sys import argv

from ultralytics import YOLO

pt_path = Path(argv[1]).resolve()
model = YOLO(str(pt_path))
onnx_tmp = Path(
    model.export(format="onnx", opset=17, imgsz=f"{int(argv[2])},{int(argv[3])}", dynamic=True, simplify=True))
onnx_out = pt_path.with_suffix(".onnx")
if onnx_tmp != onnx_out:
    onnx_tmp.replace(onnx_out)
