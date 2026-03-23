import torch, os , sys
from thop import profile, clever_format



# 1. Dynamically find the root of your project
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.benchmarks.face_alignment.ffmw_arch.options.test_options import TestOptions
from src.benchmarks.face_alignment.ffmw_arch.models import create_model

def benchmark_ffmw():
    # 1. Setup Options (Simulating the command line)
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.batch_size = 1
    
    # 2. Create the Model
    # This will load the architecture (Generator) based on your --model flag
    model = create_model(opt)
    model.setup4test(opt)
    
    # 3. Isolate the Generator
    # In pix2pix/CycleGAN, the generator is usually stored in model.netG
    netG = model.netG
    netG.eval()

    # 4. Define Input Size
    # Most frontalization models use 128x128 or 256x256
    # Check your --load_size option; we'll assume 128x128 for FFMW
    input_size = (1, 3, 128, 128)
    dummy_input = torch.randn(input_size)

    # 5. Profile
    print(f"Profiling Generator with input {input_size}...")
    macs, params = profile(netG, inputs=(dummy_input,), verbose=False)
    
    flops = macs * 2
    flops_fmt, params_fmt = clever_format([flops, params], "%.3f")

    print('\n' + '='*40)
    print(f"{'FFMW Generator Metrics':^40}")
    print('='*40)
    print(f"Parameters         : {params_fmt}")
    print(f"Total FLOPs        : {flops_fmt}")
    print(f"Model Size (est)   : {(params * 4) / (1024**2):.2f} MB")
    print('='*40 + '\n')

if __name__ == "__main__":
    # You must provide basic flags so the model knows what to build
    # Example: python benchmark_ffmw.py --model pix2pix --name experiment_1
    benchmark_ffmw()