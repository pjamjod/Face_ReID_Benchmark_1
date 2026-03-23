import torch
from thop import profile, clever_format
import sys
import os

# 1. Dynamically find the root of your project
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.benchmarks.face_detection.featherface_arch.models.retinaface import RetinaFace
from src.benchmarks.face_detection.featherface_arch.config import cfg_mnet

def calculate_flops(net, img_dim):
    # 1. Create a dummy image
    dummy_input = torch.randn(1, 3, img_dim, img_dim)
    
    # 2. Get the actual output to see the resolution/shape
    net.eval()
    with torch.no_grad():
        outputs = net(dummy_input)
    
    # RetinaFace usually returns (loc, conf, landm)
    # We'll extract the shapes of these tensors
    output_info = []
    if isinstance(outputs, tuple):
        for i, out in enumerate(outputs):
            output_info.append(f"Out_{i}: {list(out.shape)}")
    else:
        output_info.append(str(list(outputs.shape)))

    # 3. Run THOP for MACs and Params
    macs, params = profile(net, inputs=(dummy_input,), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    
    print('========================')
    print(f'Model Architecture : RetinaFace (MobileNet0.25)')
    print(f'Input Resolution   : {img_dim}x{img_dim}')
    print(f'Output Resolution  : {" | ".join(output_info)}')
    print(f'Computational MACs : {macs_fmt}')
    print(f'Parameters         : {params_fmt}')
    print('========================')

if __name__ == '__main__':
    cfg = cfg_mnet
    # Fallback to 640 if image_size isn't explicitly in the config dict
    img_dim = cfg.get('image_size', 640) 
    
    print("Building network...")
    net = RetinaFace(cfg=cfg, phase='test')
    
    print("Calculating FLOPs...")
    calculate_flops(net, img_dim)