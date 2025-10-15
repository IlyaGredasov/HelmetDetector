from pathlib import Path
from sys import argv

from ultralytics import YOLO

print(argv)
img_size = f"{argv[2]},{argv[3]}"
model = YOLO(argv[1])
onnx_model = model.export(format="onnx", opset=12, imgsz=img_size, dynamic=True, simplify=True)
onnx_path = Path(onnx_model)
