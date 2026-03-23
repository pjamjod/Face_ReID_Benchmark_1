import onnxruntime as ort
options = ort.SessionOptions()
options.enable_profiling = True
session = ort.InferenceSession("models/face_detection/scrfd10gkps_int32.onnx", options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Run one inference to generate the data
import numpy as np
input_name = session.get_inputs()[0].name
data = np.random.randn(1, 3, 640, 640).astype(np.float32) # match your input shape
session.run(None, {input_name: data})

# Stop profiling and get the filename
prof_file = session.end_profiling()
print(f"Profiling file saved to: {prof_file}")