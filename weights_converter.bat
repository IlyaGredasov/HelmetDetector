setlocal enableextensions enabledelayedexpansion

for /f "usebackq eol=# tokens=1* delims==" %%A in (".env") do (
  if not "%%~A"=="" set "%%~A=%%~B"
)

python .\onnx_weights_converter.py .\helmet_detector.pt %IMG_W% %IMG_H%

trtexec --onnx=helmet_detector.onnx --saveEngine=helmet_detector.engine ^
  --minShapes=images:%MIN_BATCH%x3x%IMG_H%x%IMG_W% ^
  --optShapes=images:%OPT_BATCH%x3x%IMG_H%x%IMG_W% ^
  --maxShapes=images:%MAX_BATCH%x3x%IMG_H%x%IMG_W% ^
  --bf16
