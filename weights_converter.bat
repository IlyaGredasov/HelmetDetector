@echo off
setlocal enableextensions enabledelayedexpansion

if "%~1"=="" (
  echo Usage: weights_converter.bat models\model.pt
  exit /b 1
)

for /f "usebackq eol=# tokens=1* delims==" %%A in (".env") do (
  if not "%%~A"=="" set "%%~A=%%~B"
)

set "MODEL_PATH=%~f1"
set "MODEL_DIR=%~dp1"
set "MODEL_NAME=%~n1"
set "ONNX_PATH=%MODEL_DIR%%MODEL_NAME%.onnx"
set "ENGINE_PATH=%MODEL_DIR%%MODEL_NAME%.engine"

python .\onnx_weights_converter.py "%MODEL_PATH%" %IMG_W% %IMG_H%

trtexec --onnx="%ONNX_PATH%" --saveEngine="%ENGINE_PATH%" ^
  --optShapes=images:%OPT_BATCH%x3x%IMG_H%x%IMG_W% ^
  --bf16 ^
  --inputIOFormats=fp16:chw ^
  --outputIOFormats=fp16:chw ^
  --memPoolSize=workspace:4096M ^
  --builderOptimizationLevel=5 ^
  --maxTactics=-1 ^
  --avgTiming=16 ^
  --allocationStrategy=static
