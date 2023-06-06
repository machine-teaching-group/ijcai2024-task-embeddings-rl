import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import torch
from torch.utils.data import Dataset
import numpy as np
from utils import load_json

class BehavioralCloningDataset(Dataset):
    def __init__(self, input_path):
        self.data = load_json(input_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state = self.data[idx]['s']
        action = self.data[idx]['a']
        
        return torch.Tensor(state).view(-1,), action
        
            
    
    
