import torch
import torch.onnx
import os
from face_detection.featherface_arch.models.retinaface import RetinaFace
from face_detection.featherface_arch.config import cfg_mnet

def export_onnx_clean():
    net = RetinaFace(cfg=cfg_mnet, phase='test')
    
    # 1. Load weights normally!
    checkpoint = torch.load('./models/face_detection/feather_face_mobilenet0.25_Final.pth', map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # strict=True ensures we actually load all the weights this time
    net.load_state_dict(clean_dict, strict=True)
    net.eval()
    print("✅ Weights loaded successfully!")

    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 2. Sanity Check
    with torch.no_grad():
        loc, conf, landm = net(dummy_input)
        print(f"Sanity Check - Max Conf Logit: {conf[:, 1].max().item():.4f}")

    # 3. Export
    torch.onnx.export(
        net,
        dummy_input,
        "featherface_640x640.onnx",
        export_params=True,
        opset_version=12,  # Try 12 first, it decomposes DCN well for older runtimes
        do_constant_folding=True,
        input_names=['input'],
        output_names=['loc', 'conf', 'landm']
    )
    print("✅ Export complete with original DCN architecture!")

if __name__ == "__main__":
    export_onnx_clean()