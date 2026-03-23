import  keras_cv_attention_models
import sys
import os
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Now you can import your usual modules
import tensorflow as tf
import keras

# 1. Dynamically find the root of your project (Benchmark_1)
# This goes up 3 levels: flops_test -> benchmarks -> src -> root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

# 2. Add the root to Python's path so it understands absolute imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. Use an absolute import starting from 'src'
from src.benchmarks.face_recognition.ghostfacenet_arch import GhostFaceNets

"""basic_model = GhostFaceNets.buildin_models("ghostnetv1", dropout=0, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4, apply_to_batch_normal=False)
basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)"""

basic_model = keras.models.load_model("models/face_recognition/GhostFaceNet_W1.3_S1_ArcFace.h5", compile=False)


from keras_cv_attention_models import model_surgery
model_surgery.get_flops(basic_model)
model_surgery.count_params(basic_model)
#model_surgery.export_onnx(basic_model, fuse_conv_bn=True, batch_size=1, simplify=True)


