## Face Detection Benchmark

### Notes

Find all the models and datasets here 
https://drive.google.com/drive/folders/1C5OmbEOIcEPVOVLLF2PnyUYg7sp2Fx3O?usp=sharing

Version thats works
- Python: 3.12
- CUDA: 12.1
- cuDNN: 9.2

Use the steps below to run the WIDER Face detection benchmark with the CUDA benchmark script.

### 0. Create a Python 3.12 virtual environment and verify the system

From the repository root, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Expected:

- `Python 3.12.x`

### 1. Download the WIDER Face dataset and evaluation code

- Download the WIDER Face dataset and the evaluation toolkit.
- Extract or copy them into `datasets/widerface/`.
- If you want to share the files from Google Drive, add your link here in the README.

The benchmark expects this structure:

```text
datasets/widerface/
	evaluation.py
	setup.py
	WIDER_val/
	wider_face_eval_tools/
```

### 1.1 Build the WIDER Face evaluation extension

From the repository root, run:

```powershell
cd datasets/widerface
python setup.py build_ext --inplace
cd ../..
```

### 2. Download the candidate face detection models

- Download the candidate ONNX models.
- Put them in `models/face_detection/`.
- If you want to share the model files from Google Drive, add your link here in the README.

The current CUDA benchmark script expects these files:

```text
models/face_detection/scrfd10gkps.onnx
models/face_detection/face_detection_yunet_2023mar_raven.onnx
```

### 3. Install Python dependencies

From the repository root, run:

```powershell
pip install -r requirements.txt
```

If you are using a virtual environment, activate it first.

### 3.1 Verify CUDA and ONNX Runtime

Run a quick runtime check:

```powershell
python -c "import torch, onnxruntime as ort; print('Torch CUDA:', torch.cuda.is_available()); print('ORT providers:', ort.get_available_providers())"
```

Expected:

- `Torch CUDA: True` (if CUDA is set up correctly)
- `ORT providers` includes `CUDAExecutionProvider`

### 4. Run the CUDA benchmark

From the repository root, run:

```powershell
python src/benchmarks/benchmark_detector_cuda.py
```

The script will prompt you to select a CUDA device if one is available.

### Outputs

The benchmark writes logs and evaluation outputs under `src/benchmarks/results/`.
