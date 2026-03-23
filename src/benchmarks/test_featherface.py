from __future__ import print_function
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import cv2
import time
import csv

# 1. --- Path Injection for Absolute Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
widerface_path = os.path.join(project_root, "datasets", "widerface")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if widerface_path not in sys.path:
    sys.path.insert(0, widerface_path)

# 2. --- Import Evaluation ---
from datasets.widerface.evaluation import evaluation

# FeatherFace Imports
from face_detection.featherface_arch.config import cfg_mnet
from face_detection.featherface_arch.layers.functions.prior_box import PriorBox
from face_detection.featherface_arch.utils.nms.py_cpu_nms import py_cpu_nms
from face_detection.featherface_arch.models.retinaface import RetinaFace
from face_detection.featherface_arch.utils.box_utils import decode, decode_landm
from face_detection.featherface_arch.utils.timer import Timer

parser = argparse.ArgumentParser(description='Retinaface Benchmark')
parser.add_argument('-m', '--trained_model', default='./models/face_detection/feather_face_mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='src/benchmarks/results/widerface_preds_featherface/', type=str, help='Dir to save txt results')
parser.add_argument('--dataset_folder', default='datasets/widerface/WIDER_val/images/', type=str, help='dataset path')
parser.add_argument('--gt_dir', default='datasets/widerface/wider_face_eval_tools/eval_tools/ground_truth', type=str, help='Ground truth mat files')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def select_device():
    """Interactive prompt to select the CUDA device; falls back to CPU."""
    print("\n" + "=" * 50)
    print("PyTorch CUDA DEVICE SELECTION")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"[{i}] {torch.cuda.get_device_name(i)}")

    print("=" * 50)

    while True:
        try:
            choice = input(f"Select a CUDA device ID [0-{num_devices - 1}]: ")
            device_id = int(choice)
            if 0 <= device_id < num_devices:
                print(f"\nSelected: {torch.cuda.get_device_name(device_id)} (ID: {device_id})")
                return torch.device(f"cuda:{device_id}")
            print("Invalid ID. Please select a valid number from the list.")
        except ValueError:
            print("Please enter a valid integer.")

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    # Safe CPU load first
    pretrained_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':
    # 0. Prompt User for Device First
    device = select_device()
    
    torch.set_grad_enabled(False)
    cfg = cfg_mnet if args.network == "mobile0.25" else None
    
    # 1. Setup Model
    net = RetinaFace(cfg=cfg, phase='test')
    
    model_path = os.path.join(project_root, args.trained_model.strip('./'))
    net = load_model(net, model_path)
    net.eval()
    print('Finished loading model!')

    # --- CUDA/CPU DEVICE SETUP ---
    net = net.to(device)
    if device.type == "cuda":
        print(f"Running on GPU via CUDA: {torch.cuda.get_device_name(device.index)}")
    else:
        print("Running on CPU.")
    # --------------------------

    # 2. Setup Paths
    testset_folder = os.path.join(project_root, args.dataset_folder.strip('./'))
    save_folder = os.path.join(project_root, args.save_folder.strip('./'))
    gt_dir = os.path.join(project_root, args.gt_dir.strip('./'))
    LOG_FILE_PATH = os.path.join(project_root, "src/benchmarks/results/inference_log_featherface.csv")
    
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    _t = {'forward_pass': Timer(), 'misc': Timer()}
    total_images_processed = 0

    # 3. Iterate through Events and Images
    print(f"Starting inference on {testset_folder}...")
    events = [d for d in os.listdir(testset_folder) if os.path.isdir(os.path.join(testset_folder, d))]

    # Open CSV logger
    with open(LOG_FILE_PATH, mode="w", newline="", encoding="utf-8") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["Event", "Image_Name", "Inference_Time_ms"])

        for event in tqdm(events, desc="Processing Events"):
            event_path = os.path.join(testset_folder, event)
            images = [img for img in os.listdir(event_path) if img.endswith(".jpg")]
            
            for img_name in images:
                image_path = os.path.join(event_path, img_name)
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if img_raw is None:
                    continue
                    
                img = np.float32(img_raw)
                total_images_processed += 1

                # Scale and preprocessing
                target_size = 1600
                max_size = 2150
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                resize = float(target_size) / float(im_size_min)

                if np.round(resize * im_size_max) > max_size:
                    resize = float(max_size) / float(im_size_max)
           
                if args.origin_size:
                    resize = 1

                if resize != 1:
                    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                    
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                
                # Send data to selected CUDA/CPU device
                img = img.to(device)
                scale = scale.to(device)

                # --- START INFERENCE TIMER ---
                start_time = time.perf_counter()

                _t['forward_pass'].tic()
                loc, conf, landms = net(img)  # forward pass
                _t['forward_pass'].toc()
                
                _t['misc'].tic()
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward().to(device)
                prior_data = priors.data
                
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

                # ignore low scores
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1]
                boxes = boxes[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, args.nms_threshold)
                dets = dets[keep, :]
                _t['misc'].toc()
                
                # --- STOP INFERENCE TIMER ---
                end_time = time.perf_counter()
                inf_time_ms = (end_time - start_time) * 1000 
                
                # Log to CSV
                csv_writer.writerow([event, img_name, round(inf_time_ms, 2)])

                # 4. Save to WiderFace .txt format
                txt_name = img_name.replace(".jpg", ".txt")
                save_name = os.path.join(save_folder, event, txt_name)
                os.makedirs(os.path.dirname(save_name), exist_ok=True)

                with open(save_name, "w") as fd:
                    fd.write(img_name.replace(".jpg", "") + "\n")
                    fd.write(str(len(dets)) + "\n")
                    for box in dets:
                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2]) - int(box[0])
                        h = int(box[3]) - int(box[1])
                        confidence = str(box[4])
                        fd.write(f"{x} {y} {w} {h} {confidence}\n")

    # 5. Calculate FPS
    if total_images_processed > 0:
        total_time = _t['forward_pass'].total_time + _t['misc'].total_time
        fps = total_images_processed / total_time
        print(f'\nProcessed {total_images_processed} images.')
        print(f'Average FPS: {fps:.2f}')
        print(f"Log saved to: {LOG_FILE_PATH}")
    
    # 6. Run Evaluation
    print("\nInference complete. Starting Evaluation...")
    evaluation(save_folder, gt_dir)