from ptflops import get_model_complexity_info
import argparse
import torch
import sys
import os

# 1. Dynamically find the root of your project (Benchmark_1)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.benchmarks.face_recognition.transface_arch.backbones import get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n', type=str, default="vit_s_dp005_mask_0")
    args = parser.parse_args()
    
    # 1. Load Model
    net = get_model(args.n)
    net.eval() # Set to evaluation mode
    
    # 2. Define Input Shape (Batch Size of 1)
    input_shape = (1, 3, 112, 112)
    dummy_input = torch.randn(input_shape)
    
    # 3. Quick Forward Pass to get Output Shape
    with torch.no_grad():
        # Handle autocast warning by using the modern syntax if needed, 
        # or just run it normally.
        output = net(dummy_input)
        
        # Some models return tuples (e.g., embeddings + classifications). 
        # This handles both single tensors and tuples.
        if isinstance(output, tuple) or isinstance(output, list):
            output_shape = [tuple(o.shape) for o in output]
        else:
            output_shape = tuple(output.shape)

    # 4. Print I/O Summary
    print("\n" + "="*40)
    print(f"Model  : {args.n}")
    print(f"Input  : {input_shape}")
    print(f"Output : {output_shape}")
    print("="*40 + "\n")

    # 5. Calculate Complexity (ptflops expects shape without batch dimension)
    macs, params = get_model_complexity_info(
        net, (3, 112, 112), as_strings=False,
        print_per_layer_stat=False, verbose=True)
        
    gmacs = macs / (1000**3)
    
    print("\n" + "="*40)
    print(f"Complexity : {gmacs:.3f} GFLOPs") # Note: ptflops returns MACs, but names the variable MACs. To get true FLOPs, usually you multiply by 2.
    print(f"Parameters : {params/(1000**2):.3f} M")

    if hasattr(net, "extra_gflops"):
        print(f"Extra-GFLOPs: {net.extra_gflops:.3f}")
        print(f"Total-GFLOPs: {gmacs + net.extra_gflops:.3f}")
    print("="*40 + "\n")