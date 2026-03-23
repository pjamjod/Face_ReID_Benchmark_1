import os
import os.path as osp
import shutil
import ffmpeg
import cv2
import glob
import re
import hashlib
from typing import Tuple, Union
import numpy as np
import base64
from io import BytesIO
from tqdm import tqdm

def atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text

def natural_keys(text) -> list:
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def clear_dir(dir: str):
    remove_dir(dir)
    os.makedirs(dir)

def remove_dir(dir: str):
    try:
        shutil.rmtree(dir)
    except:
        pass

def merge_dirs(source_dir: str, target_dir: str, remove_source: bool = False):
    for file_name in tqdm(os.listdir(source_dir), desc="Merging directories"):
        shutil.move(osp.join(source_dir, file_name), osp.join(target_dir, file_name))

def dict_to_text(dictionary: dict) -> str:
    text = ''
    for key, value in dictionary.items():
        key = '- ' + key.capitalize()
        text += f"{key}: {value}\n\n"
    return text
    
import re

def check_and_format_cientific_notation(input_string: str) -> Union[float, str]:
    # Regular expression pattern to match scientific notation
    pattern = r'[-+]?\d*\.?\d+e[-+]?\d+'
    
    # Find all matches in the input string
    match = re.findall(pattern, input_string)
    if match:
        return float(float(match[0]))
    else:
        return input_string

def generate_md5_hash(file_path: str) -> str:
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        # Create an MD5 hash object
        md5_hash = hashlib.md5()
        
        # Read the file in chunks and update the hash
        while chunk := file.read(8192):
            md5_hash.update(chunk)
    
    # Return the hexadecimal representation of the digest
    return md5_hash.hexdigest()

def compute_magnitude_units(params: int) -> str:
        # Extract number of params
    if params >= 1_000_000_000:
        params = f"{params / 1_000_000_000:.1f}B"
    elif params >= 1_000_000:
        params = f"{params / 1_000_000:.1f}M"
    elif params >= 1_000:
        params = f"{params / 1_000:.1f}K"
    else:
        params = str(params)
    return params

def compute_mem_size_units(bytes: int) -> str:
    if bytes >= 1_000_000_000:
        bytes = f"{bytes / 1_000_000_000:.2f}GB"
    elif bytes >= 1_000_000:
        bytes =  f"{bytes / 1_000_000:.2f}MB"
    elif bytes >= 1_000:
        bytes =  f"{bytes / 1_000:.2f}KB"
    else:
        bytes = f"{bytes}B"
    return bytes

def image_to_base64(image:np.array, plot_title: str, plot_size: int) -> str:
    is_success, image_buffer = cv2.imencode(".png", image)
    buf = BytesIO(image_buffer)
    buf.seek(0)
    png_data = buf.read()
    base64_utf8_str = base64.b64encode(png_data).decode('utf-8')
    dataurl = f'data:image/png;base64,{base64_utf8_str}'
    image_md = '<img src="'+dataurl+'" alt="'+plot_title+'" width="'+str(plot_size)+'"/>'
    #image_md += f'<p>{plot_title}</p>'
    return image_md

## Video processing functions
def get_fps(input_video_path: str) -> int:
    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def extract_video_frames(video_path: str, output_path: str):
    try:
        input_stream = ffmpeg.input(video_path)
        video = input_stream.video
        output_pattern = os.path.join(output_path, 'frame_%04d.jpg')
        ffmpeg.output(video, output_pattern, start_number=0).run(overwrite_output=True)
        print("Frames extracted successfully.")
    except ffmpeg.Error as e:
        print(f"An error occurred:\n{e}")

def extract_video_audio(video_path: str, output_path: str):
    try:
        input_stream = ffmpeg.input(video_path)
        audio = input_stream.audio
        output_file = os.path.join(output_path, 'audio.wav')
        ffmpeg.output(audio, output_file).run(overwrite_output=True)
        print("Audio extracted successfully.")
    except ffmpeg.Error as e:
        print(f"An error occurred:\n{e}")

def video_from_frames(frames_path: str, output_path: str, fps: int, video_name:str ='pred_video.mp4'):
    try:
        input_pattern, start_number = find_ffmpeg_input_pattern_and_start_number(frames_path)
        input_pattern = os.path.join(frames_path, input_pattern)
        output_file = os.path.join(output_path, video_name)
        ffmpeg.input(input_pattern, framerate=fps, start_number=start_number).output(output_file).run(overwrite_output=True)
        print("Video recovered successfully.")
    except ffmpeg.Error as e:
        print(f"An error occurred:\n{e}")


def update_video_frames(video_path: str, frames_path: str, output_path: str):
    try:    
        video_name = os.path.basename(video_path)
        output_file = os.path.join(output_path, 'pred_'+video_name)
        fps = get_fps(video_path)
        input_pattern = os.path.join(frames_path, "frame_%04d.jpg")
        input_stream_1 = ffmpeg.input(input_pattern, framerate=fps, start_number=0)
        input_stream_2 =ffmpeg.input(video_path)
        ffmpeg.concat(input_stream_1, input_stream_2, v=1, a=1).output(output_file, vcodec="libx264").overwrite_output().run()
        print("Frames updated successfully.")
    except ffmpeg.Error as e:
        print(f"An error occurred:\n{e}")


def find_ffmpeg_input_pattern_and_start_number(folder: str) -> Union[Tuple[str, int], Tuple[None, None]]:
    files = os.listdir(folder)
    # Filter to include only files (exclude directories)
    files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
    if not files:
        print("No files found in the directory.")
        return None, None

    # Split filenames into sequences of digits and non-digits
    tokens_list = []
    for f in files:
        tokens = re.findall(r'\D+|\d+', f)
        tokens_list.append(tokens)

    # Check that all filenames split into the same number of tokens
    num_tokens = len(tokens_list[0])
    for tokens in tokens_list:
        if len(tokens) != num_tokens:
            print("Filenames do not have the same pattern.")
            return None, None

    # Determine the pattern for each token position and collect numbers
    pattern_tokens = []
    start_numbers = []
    numeric_token_indices = []
    for i in range(num_tokens):
        token_samples = [tokens[i] for tokens in tokens_list]

        if all(re.match(r'\D+', t) for t in token_samples):
            # Non-digit token, ensure all are identical
            if len(set(token_samples)) == 1:
                pattern_tokens.append(token_samples[0])
            else:
                print(f"Non-digit tokens at position {i} do not match.")
                return None, None
        elif all(re.match(r'\d+', t) for t in token_samples):
            # Digit token, collect numbers
            numbers = [int(t) for t in token_samples]
            start_number = min(numbers)
            start_numbers.append(start_number)
            numeric_token_indices.append(i)

            # Determine if zero-padded
            lengths = [len(t) for t in token_samples]
            min_len = min(lengths)
            max_len = max(lengths)
            zero_padded = all(len(t) == max_len for t in token_samples)

            if zero_padded and min_len == max_len:
                N = max_len
                pattern_tokens.append(f'%0{N}d')
            else:
                pattern_tokens.append('%d')
        else:
            print(f"Token mismatch at position {i}.")
            return None, None

    # Construct the FFmpeg input pattern
    pattern = ''.join(pattern_tokens)

    if len(start_numbers) == 1:
        start_number = start_numbers[0]
    else:
        print("Multiple numeric sequences found. FFmpeg may not support multiple '%d' tokens.")
        start_number = None

    return pattern, start_number

