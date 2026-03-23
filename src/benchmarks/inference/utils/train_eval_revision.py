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


class TrainEvalRevision(GenericRevision):
    def __init__(self):
        super().__init__()
        self.train_losses = pd.DataFrame()
        self.validation_losses = pd.DataFrame()

        self.train_metrics = pd.DataFrame()

        self.train_lr = pd.DataFrame()

        self.validation_class_metrics = pd.DataFrame()
        self.test_class_metrics = pd.DataFrame()

        self.md_structure = """# Experiment report\n\n## Results\n {}\n\n\n## Comments\n\n\n"""
        self.report_name = 'revision'


    # TODO: Implement the following method
    def load_params(self, params: dict):
        pass


    def generate_report(self, data_path: str):
        train_path = osp.join(data_path, 'train')
        eval_path = osp.join(data_path, 'test')

        assert osp.exists(train_path), 'Train data path does not exist'
        assert osp.exists(eval_path), 'Evaluation data path does not exist'

        train_metrics, train_plots = self.load_data(train_path)
        test_metrics, test_plots = self.load_data(eval_path)

        metrics = train_metrics
        metrics.update(test_metrics)
        self.tables = metrics
        plots_metrics = train_plots
        plots_metrics.update(test_plots)
        self.plots = self.generate_plots(plots_metrics)

        report_list = self.plots_to_base64(self.plots)
        report_list.update(self.tables_to_md(metrics))

        self.md_report = self.md_structure.format(''.join(list(report_list.values())))
        

    def load_data(self, data_path: str) -> Tuple[dict, dict]:

        files = os.listdir(data_path)

        plots_data = {}
        metrics_data = {}
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
                if 'class' in file_name:
                    metrics_data[file_name] = data
                else:
                    plots_data[file_name] = data
            elif file == 'train_time.json':
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    json_data = json.load(f)
                    json_data_as_lists = {key: [value] if not isinstance(value, list) else value for key, value in json_data.items()}
                    metric_pd = pd.DataFrame.from_dict(json_data_as_lists)
                    metrics_data[file_name] = metric_pd
        return metrics_data, plots_data


    def merge_dict_by_root_key(self, data: dict, exclude_words:list =['train', 'validation', 'test']) -> dict:
        merged_dict = defaultdict(dict)
        for key, value in data.items():
            key_parts = key.split('_')
            prefix = next((part for part in key_parts if part in exclude_words), '-')
            root_key_parts = [part for part in key_parts if part not in exclude_words]
            root_key = '_'.join(root_key_parts)
            merged_dict[root_key][prefix] = value

        return dict(merged_dict)
