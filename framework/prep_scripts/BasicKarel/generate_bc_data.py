import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.BasicKarel import BasicKarelEnv
import numpy as np
import random
from tqdm import tqdm
from glob import glob
from utils import load_json, dump_json 

def generate_data():
    act_map = {'move': 0, 'turnLeft': 1, 'turnRight': 2, 'pickMarker': 3, 'putMarker': 4, 'finish': 5}
    
    data_size = len(glob('envs/data/basic_karel/data/train/task/*.json'))
    env = BasicKarelEnv(failure_gamma=1)
    
    for i in range(4):
        # (i == 0): Default, (i == 1): Mask Pick, (i == 2): Mask Put, (i == 3): Mask Pick and Put
        if i == 0:
            masked_actions = []
        elif i == 1:
            masked_actions = [act_map['pickMarker']]    
        elif i == 2:
            masked_actions = [act_map['putMarker']]   
        elif i == 3:
            masked_actions = [act_map['pickMarker'], act_map['putMarker']]        
    
        examples = []
    
        for idx in tqdm(range(data_size)):
            seq_data = load_json(f'envs/data/basic_karel/data/train/seq/{idx}_seq.json')
            task_data = load_json(f'envs/data/basic_karel/data/train/task/{idx}_task.json')
        
            state = env.reset(state=task_data)
            count = 0
            while True:
                example = {}
                example['s'] = state.tolist()
                example['a'] = act_map[seq_data['sequence'][count]]
            
                if example['a'] not in masked_actions:
                    examples.append(example)
            
                if seq_data['sequence'][count] == 'finish':
                    break
            
                state, _, _, _ = env.step(example['a'])
                count += 1
        
        if i == 0:
            dump_json(examples, 'embedding_network/bc_data/BasicKarel_default_data.json')
        elif i == 1:
            dump_json(examples, 'embedding_network/bc_data/BasicKarel_mask_pick_data.json')   
        elif i == 2:
            dump_json(examples, 'embedding_network/bc_data/BasicKarel_mask_put_data.json')    
        elif i == 3:
            dump_json(examples, 'embedding_network/bc_data/BasicKarel_mask_pick_put_data.json')     

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    generate_data()             