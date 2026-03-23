from pynvml import *
import threading
import math
import numpy as np
from typing import Tuple,Union

class GpuMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.gpu_usage = []
        self.gpu_memory_usage = []
        self.reset_flag = False
        self.stop_event = threading.Event()

        self.gpu_name, self.gpu_available = self.check_gpu_availabilty()

        print('GPU monitor running')

    def run(self):
        if self.gpu_available:
            while not self.stop_event.is_set():
                if self.reset_flag:
                    self.gpu_usage = []
                    self.gpu_memory_usage = []
                    self.reset_flag = False
                h = nvmlDeviceGetHandleByIndex(0)
                mem_res = nvmlDeviceGetMemoryInfo(h).used/math.pow(1024, 2) # In MiB
                usage_res = nvmlDeviceGetUtilizationRates(h).gpu
                self.gpu_memory_usage.append(mem_res)
                self.gpu_usage.append(usage_res)
        else:
            print('GPU not available')

    def stop(self):
        self.stop_event.set()
        print('Stopping GPU Monitor')
        self.join()
        print('GPU Monitor stopped')
    
    def check_gpu_availabilty(self) -> Tuple[Union[str, bool], bool]:
        nvmlInit()
        if nvmlDeviceGetCount()>0:
            h = nvmlDeviceGetHandleByIndex(0)
            gpu_name = nvmlDeviceGetName(h)
            return gpu_name, True
        else:
            return None, False

    def reset(self):
        self.reset_flag = True

    def get_data(self) -> dict:
        gpu_merics = {"gpu_name": self.gpu_name, 
                      "gpu_usage": str(max(self.gpu_usage)) + '%', 
                      "gpu_memory_usage": str(max(self.gpu_memory_usage)) + 'MiB'}
        return gpu_merics