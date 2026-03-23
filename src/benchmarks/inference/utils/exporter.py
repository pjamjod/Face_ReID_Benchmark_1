from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os

from ..utils.utils import clear_dir, dict_to_text
from .inference import GenericInference
from .params import Params
from ..onnx.onnx_inspector import extract_onnx_model_stats

class GenericExporter:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.params = Params()

        self.exporter = GenericInference()
        self.exported_model_path = None
        self.export_format = None
        self.fixed_inputs_shape = {}

        self.model_info = None
        self.dataset_info = None
        self.exported_model_info = None
        self.model_performance_info = None

        self.model_card_template = "# Model Card\n## Model Overview\n{}\n\n## Dataset\n{}\n## Train Performance\n{}\n## Benchmark Performance\n\n## Contact\n\n"
        self.model_card = None

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def load_model(self, models_path: str):
        pass

    @abstractmethod
    def export_onnx_model(self, models_path: str, name: str ='best_model'):
        pass

    @abstractmethod
    def check_pretrained(self):
        pass

    def load_params(self, params: dict):
        for key, value in params.items():
            for sub_key, sub_value in value.items():
                setattr(self.params, key+'_'+sub_key, sub_value)

    def generate_model_card(self):
        self.extract_model_info()
        self.extract_model_performance()
        self.extract_dataset_info()
        self.extract_exported_model_info()
        
        model_overview = self.model_info
        model_overview.update(self.exported_model_info)

        model_overview = model_overview if model_overview is not None else {}
        model_performance_info = self.model_performance_info if self.model_performance_info else {}
        dataset_info = self.dataset_info if self.dataset_info else {}

        self.model_card = self.model_card_template.format(dict_to_text(model_overview), dict_to_text(dataset_info), dict_to_text(model_performance_info))

    def save_model_card(self, model_save_path: str, name: str ='best_onnx_model_card.md'):
        model_card_path = osp.join(model_save_path, name)

        with open(model_card_path, 'w') as file:
            file.write(self.model_card)
    
    def extract_exported_model_info(self):
        if self.export_format == 'onnx':
            self.exported_model_info = extract_onnx_model_stats(self.exported_model_path, self.fixed_inputs_shape)
        else:
            raise NotImplementedError('Extract exported model info for {} is not implemented'.format(self.export_format))
    
