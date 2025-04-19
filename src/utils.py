import os
import torch

def save_model(model,filepath):
    torch.save(model,filepath)

def load_model(filepath):
    return torch.load(filepath,weights_only=False)