from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os
import random
import numpy as np

from ..utils.utils import clear_dir
from ..metrics.evaluate_metrics import TestMetrics
from .inference import GenericInference
from .params import Params


class GenericEvaluator:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.params = Params()
        self.metrics = TestMetrics()
        self.test_data = None
        self.run_path = None
        self.evaluated = False
        self.classes_dict = {}
        self.inferer = GenericInference()

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def load_data(self, dataset: str):
        pass

    @abstractmethod
    def extract_metrics(self) -> TestMetrics:
        pass

    @abstractmethod
    def load_model(self, models_path: str):
        pass

    @abstractmethod
    def save_predictions(self, save_path: str):
        pass

    def load_params(self, params: dict):
        for key, value in params.items():
            for sub_key, sub_value in value.items():
                setattr(self.params, key+'_'+sub_key, sub_value)

    def set_random_seed(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def inference(self, data_path: str, params: dict, save_path: str):
        self.inferer.load_params(params)
        self.inferer.load_model(self.model_path)
        self.inferer.predict(data_path, save_path)

    def export_evaluate_results(self, save_path: str):
        clear_dir(save_path)
        if self.evaluated:
            metrics = self.extract_metrics()
        else:
            metrics = self.metrics
        metrics.save_to_file(save_path)

    def clear(self):
        try:
            shutil.rmtree(self.run_path)
        except:
            pass
