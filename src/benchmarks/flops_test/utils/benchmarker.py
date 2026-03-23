from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os
from collections import defaultdict

from ..utils.utils import clear_dir, generate_md5_hash, dict_to_text
from .params import Params
from ..metrics.benchmark_metrics import BenchmarkMetrics
from ..utils.benchmark_revision import BenchmarkRevision

class GenericBenchmarker():
    def __init__(self):
        self.model_path = None
        self.params = Params()
        self.metrics = BenchmarkMetrics()
        self.data = None

        self.run_path = 'runs'

        self.reviewer = BenchmarkRevision()


    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self, dataset_path: str):
        pass

    def load_params(self, params: dict):
        for key, value in params.items():
            for sub_key, sub_value in value.items():
                setattr(self.params, key+'_'+sub_key, sub_value)

        #self.reviewer.load_params(params)
        inf_params = params['benchmark']
        # TODO revise self.params.__dict__ conversion
        self.inferer.load_params(inf_params)

    def generate_model_info(self) -> dict:
        model_info = {
            'model_path': self.model_path,
            'model_hash': generate_md5_hash(self.model_path),
            'dataset': osp.basename(self.data.dataset_name),
            'folder': os.getcwd()
        }
        return model_info

    def save_benchmark(self, save_path: str):
        
        self.metrics.save_to_file(save_path)
    
        self.reviewer.generate_report(save_path)
        self.reviewer.save_report(save_path, 
                                  save_plots=False,
                                  save_tables=True, 
                                  single_file=True, 
                                  save_pdf=False, 
                                  save_html=False)
    
        if self.run_path != save_path:
            shutil.move(self.run_path, save_path, copy_function = shutil.copytree)
    
    def update_model_card(self):
        # TODO: Find a more rebust way to update the model card, without harcoded model card name template (the model card name template is set in the exporter)
        model_card_path = osp.join(osp.dirname(self.model_path), osp.basename(self.model_path).replace('.onnx', '_onnx_model_card.md'))
        assert osp.exists(model_card_path), 'Model card does not exist: {}'.format(model_card_path)

        metrics_md = dict_to_text(self.metrics.benchmark_metrics)+'\n'
        with open(model_card_path, 'r') as f:
            lines = f.readlines()
            contact_line = None
            for i, line in enumerate(lines):
                if '## Benchmark Performance' in line:
                    benchmark_line = i
                if '## Contact' in line:
                    contact_line = i
            if contact_line is not None:
                lines = lines[:benchmark_line+2]+lines[contact_line:]
            lines.insert(benchmark_line+1, metrics_md)
        with open(model_card_path, 'w') as f:
            f.writelines(lines)