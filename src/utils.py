import os, sys
import json, time
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path_prefix = './params'

def check_convention(name):
    for a in ['knn', 'random_forest', 'svm']:
        if a in name:
            return True
    return False


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def get_avg(self):
        return self.avg

    def get_num(self):
        return self.cnt
    
class HSIRecoder(object):
    def __init__(self) -> None:
        self.record_data = {}
        self.pred = None

    def append_index_value(self, name, index, value):
        """
        index : int, 
        value: Any
        save to dict
        {index: list, value: list}
        """
        if name not in self.record_data:
            self.record_data[name] = {
                "type": "index_value",
                "index":[],
                "value":[]
            } 
        self.record_data[name]['index'].append(index)
        self.record_data[name]['value'].append(value)
    
    def record_time(self, time):
        self.record_data['eval_time'] = time

    def record_param(self, param):
        self.record_data['param'] = param 

    def record_eval(self, eval_obj):
        self.record_data['eval'] = eval_obj

    def record_pred(self, pred_matrix):
        self.pred = pred_matrix
        
    def to_file(self, path):
        time_stamp = int(time.time())
        save_path_json = "%s_%s.json" % (path, str(time_stamp))
        save_path_pred = "%s_%s.pred.npy" % (path, str(time_stamp))

        ss = json.dumps(self.record_data, indent=4)
        with open(save_path_json, 'w') as fout:
            fout.write(ss)
            fout.flush()
        np.save(save_path_pred, self.pred)

        print("save record of %s done!" % path)
        
    def reset(self):
        self.record_data = {}
        

# global recorder
recorder = HSIRecoder()
