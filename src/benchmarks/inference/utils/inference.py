from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os
import numpy as np
import cv2
from typing import Tuple, Union
import time

from .utils import clear_dir, extract_video_frames, natural_keys, update_video_frames, video_from_frames
from .params import Params
from .gpu_monitor import GpuMonitor

class GenericInference:
    def __init__(self):
        self.model_path = None

        self.data = None
        self.data_path = None
        self.data_type = None

        self.params = Params()
        
        self.save_path = None
        self.run_path = None
        self.txt_path =  None
        self.json_path = None

        self.video_extensions = ['.mp4', '.avi', '.mov']
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.color_pallete=[(242,113,51),(152,253,0),(106,23,255),(21,249,249),(249,21,211),(43,0,255)] # Color ir BGR

        self.inference_time = []
        self.gpu_metrics = {}

        self.classes_dict = None

        self.gpu_monitor = GpuMonitor()

        self.inference_metrics_file = 'inference_metrics.json'


    @abstractmethod
    def predict(self, data_path: str, save_path: str) -> list:
        pass
    
    @abstractmethod
    def load_model(self, model_path: str):
        pass
    
    def load_params(self, params: Union [dict, Params]):
        if isinstance(params, Params):
            params = params
        else:
            for key, value in params.items():
                setattr(self.params, key, value)
          
    def preprocess_data(self, data_path: str):
        self.data_path = data_path
        if osp.isdir(data_path):
            print('Inference data is a folder')
            data = [osp.join(data_path, file) for file in os.listdir(data_path)]
            data.sort(key=natural_keys)
            self.data_type = 'folder'
        else:
            _, file_extension = osp.splitext(data_path)
            if file_extension in self.video_extensions:
                print('Inference data is a video')
                frames_path = osp.join(self.run_path, 'frames')
                clear_dir(frames_path)
                extract_video_frames(data_path, frames_path)
                data = [osp.join(frames_path, file) for file in os.listdir(frames_path)]
                data.sort(key=natural_keys)
                self.data_type = 'video'
            elif file_extension in self.image_extensions:
                print('Inference data is an image')
                data = [data_path]
                self.data_type = 'image'
            else:
                raise ValueError('Data should be either a video, an image or a directory containing images, accepted extensions are:{self.video_extensions} and {self.image_extensions}')
        self.data = data

    def postprocess_data(self, frames_path: str):
        if self.params.video_out:
            if self.data_type == 'video':
                update_video_frames(self.data_path, frames_path, self.save_path)
            elif self.data_type == 'folder':
                video_name = 'pred_'+osp.basename(self.data_path)+'.mp4'
                video_from_frames(frames_path, self.save_path, self.params.fps, video_name=video_name)
        if self.run_path != self.save_path:
            if self.params.save_pred:
                shutil.move(frames_path, self.save_path)
            if self.params.save_txt:
                shutil.move(osp.join(self.run_path,self.inference_path,self.txt_path), self.save_path)
            if self.params.save_json:
                shutil.move(osp.join(self.run_path,self.inference_path,self.json_path), self.save_path)

    def get_color (self, class_id: int) -> Tuple[int, int, int]:
        if len(self.color_pallete)<=class_id:
            rng = np.random.default_rng(3)
            self.color_pallete += [(x[0],x[1],x[2]) for x in rng.uniform(0, 255, size=(class_id-len(self.color_pallete)+1, 3))]
        color = self.color_pallete[class_id]
        return color

    def plot_prediction(self, image: np.array, predictions: np.array) -> np.array:
        ''' predictions: array with the following format [(coco-xyhw), class_id, conf]'''
        if predictions.shape[0]==0:
            return image
        for k,pred in enumerate(predictions):
            x, y, w, h, class_id, conf = pred
            class_id = int(class_id)
            if self.params.draw_cls:
                if self.classes_dict!=None and class_id in self.classes_dict.keys():
                    color_id = list(self.classes_dict).index(class_id)
                    color = self.get_color(color_id)
                    class_text=self.classes_dict[class_id]
                else:
                    color = self.get_color(class_id)
                    class_text=class_id
            else:
                class_text=''

            if self.params.draw_conf:
                class_text += '-' if class_text!='' else '' 
                class_text += str(round(conf*100,2)) + '% '
            
            cv2.rectangle(image, (int(x - w // 2), int(y - h // 2)), (int(x + w // 2), int(y + h // 2)), color, self.params.box_thickness)
            if self.params.draw_conf or self.params.draw_cls:
                cv2.rectangle(image, (int(x - w // 2), int(y - h // 2)), (int(x - w // 2 + 13*len(class_text)), int(y - h // 2 - 30)), color, -1)
                cv2.putText(image, class_text, (int(x - w // 2), int(y - h // 2 - 10)), cv2.FONT_HERSHEY_DUPLEX , 0.75, (255, 255, 255), 2)

        return image

    def construct_inference_metrics(self) -> dict:
        metrics = {'inference_time': np.mean(self.inference_time)}
        metrics.update(self.gpu_metrics)
        return metrics
    
    def save_inference_metrics(self, save_path: str):
        # TODO Revise warmup time effect on inference time
        metrics = self.construct_inference_metrics()
        with open(osp.join(save_path, self.inference_metrics_file), 'w') as file:
            json.dump(metrics, file)

    def get_inference_metrics(self) -> dict:
        return self.construct_inference_metrics()

    def clear(self):
        if not self.save_path.startswith(self.run_path):
            print('Clearing run path')
            try:
                shutil.rmtree(self.run_path)
            except:
                pass