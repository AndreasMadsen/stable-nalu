
import os
import os.path as path
import torch

THIS_DIR = path.dirname(path.realpath(__file__))

if 'SAVE_DIR' in os.environ:
    SAVE_DIR = os.environ['SAVE_DIR']
else:
    SAVE_DIR = path.join(THIS_DIR, '../../save')

def save_model(name, model):
    save_file = path.join(SAVE_DIR, name) + '.pth'
    os.makedirs(path.dirname(save_file), exist_ok=True)
    torch.save(model, save_file)
