from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os
import random
import numpy as np

from ..utils.utils import clear_dir, check_and_format_cientific_notation
from .params import Params
from ..metrics.train_metrics import TrainMetrics


class GenericTrainer:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_info = {}
        self.params = Params()
        self.metrics = TrainMetrics()
        self.train_data = None
        self.train_dataset = None
        self.trained = False
        self.run_path = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def load_data(self, dataset: str):
        pass

    @abstractmethod
    def extract_metrics(self) -> TrainMetrics:
        pass

    @abstractmethod
    def load_model(self, models_path: str):
        pass

    @abstractmethod
    def save_model(self, models_path: str, name: str ='best_model'):
        pass

    @abstractmethod
    def save_model_info(self, models_path: str, name: str ='best_model'):
        pass

    @abstractmethod
    def save_predictions(self, save_path: str):
        pass

    @abstractmethod
    def extract_model_info(self) -> dict:
        pass

    def set_random_seed(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def load_params(self, params: dict):
        for key, value in params.items():
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str):
                    sub_value = check_and_format_cientific_notation(sub_value)
                setattr(self.params, key+'_'+sub_key, sub_value)

    def export_train_results(self, save_path: str):
        clear_dir(save_path)
        if self.trained:
            metrics = self.extract_metrics()
        else:
            metrics = self.metrics
        metrics.save_to_file(save_path)

    def update_model_info(self):
        new_model_info = self.extract_model_info()
        if new_model_info['info'] == {}:
            self.model_info['info'] = new_model_info['info']
        else:
            self.model_info['info'].update(new_model_info['info'])
        
        if new_model_info['performance'] == {}:
            self.model_info['performance'] = new_model_info['performance']
        else:    
            self.model_info['performance'].update(new_model_info['performance'])
            
    def clear(self):
        try:
            shutil.rmtree(self.run_path)
        except:
            pass