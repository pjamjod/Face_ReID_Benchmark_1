from abc import ABC, abstractmethod
import pandas as pd
import os.path as osp
import json
import shutil
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import base64
from typing import Tuple

from ..utils.utils import clear_dir
from ..utils.generic_revision import GenericRevision
from .params import Params


class BenchmarkRevision(GenericRevision):
    def __init__(self):
        super().__init__()
        self.train_losses = pd.DataFrame()
        self.validation_losses = pd.DataFrame()

        self.report_name = 'benchmark'
        self.md_structure = '# Benchmark\n\nModel: {}\n\nModel hash: {}\n\nDataset: {}\n\nFolder: {}\n\n\n## Metrics: {}\n\n'



    # TODO: Implement the following method
    def load_params(self, params: dict):
        pass


    def generate_report(self, data_path: str):
        assert osp.exists(data_path), 'Benchmark data path does not exist'

        metrics, model_info = self.load_data(data_path)
        self.tables = metrics
        report_list=self.tables_to_md(metrics)
        self.md_report = self.md_structure.format(model_info['model_path'], 
                                                  model_info['model_hash'],
                                                  model_info['dataset'],
                                                  model_info['folder'],
                                                  ''.join(list(report_list.values())))


    def load_data(self, data_path: str) -> Tuple[dict, dict]:

        files = os.listdir(data_path)

        metrics_data = {}
        model_info = {}
        for file in files:
            file_path = osp.join(data_path, file)
            file_name, file_extension = osp.splitext(file)
            if file_extension == '.csv':
                try:
                    data = pd.read_csv(file_path)
                except:
                    data = pd.DataFrame()
                if data.empty:
                    continue
                metrics_data[file_name] = data
            elif file == 'model_info.json':
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    model_info = json.load(f)
        assert model_info != {}, 'Model info not found or empty, please check the model_info.json file'
        return metrics_data, model_info

