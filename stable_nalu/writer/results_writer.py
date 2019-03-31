
import os
import json
import torch
import os.path as path

THIS_DIR = path.dirname(path.realpath(__file__))
RESULTS_DIR = path.join(THIS_DIR, '../../results')

class _SpecialTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return super().default(obj)

class ResultsWriter:
    def __init__(self, name):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.fp = open(path.join(RESULTS_DIR, name + '.ndjson'), 'a')

    def add(self, data):
        self.fp.write(json.dumps(data, cls=_SpecialTorchEncoder) + '\n')
        self.fp.flush()

    def __del__(self):
        self.fp.close()
