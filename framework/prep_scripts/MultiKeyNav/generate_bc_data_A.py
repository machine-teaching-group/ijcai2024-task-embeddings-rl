import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.MultiKeyNav import MultiKeyNavEnv
from envs.MultiKeyNavA import MultiKeyNavAEnv
from envs.MultiKeyNavAB import MultiKeyNavABEnv
import numpy as np
import random
from tqdm import tqdm
from utils import dump_json

def generate_data():
    # Indices: A:1, B:2, C:3, D:4
    s_idx_A = 1
    s_idx_B = 2
    s_idx_C = 3
    s_idx_D = 4
        
    # Segments: A: 0-0.1, B: 0.2-0.3, C: 0.4-0.5, D: 0.6-0.7
    seg_A = [0, 0.1]
    seg_B = [0.2, 0.3]
    seg_C = [0.4, 0.5]
    seg_D = [0.6, 0.7]
        
    # Pick Action Indices: A:2, B:3, C:4, D:5
    idx_A = 2
    idx_B = 3
    idx_C = 4
    idx_D = 5
    
    envs = [['MultiKeyNavA_default', MultiKeyNavAEnv(), []],
            ['MultiKeyNavA_maskA', MultiKeyNavAEnv(), [idx_A]],
            ['MultiKeyNavA_maskB', MultiKeyNavAEnv(), [idx_B]],
            ['MultiKeyNavA_maskC', MultiKeyNavAEnv(), [idx_C]],
            ['MultiKeyNavA_maskD', MultiKeyNavAEnv(), [idx_D]],
            ['MultiKeyNavA_maskAllKeys', MultiKeyNavAEnv(), [idx_A, idx_B, idx_C, idx_D]]]
    
    for name, env, masked_actions in envs:
        examples = []
    
        for _ in tqdm(range(100000)):
            generated = False
            while not generated:
                example = {}
                state = env.reset().tolist()
                example['s'] = state
        
                key_1_index = None
                key_2_index = None
                req_keys = None
                seg_1 = None
                seg_2 = None
                a_idx_1 = None
                a_idx_2 = None
        
                key_1_index, key_2_index = s_idx_A, s_idx_A
                seg_1, seg_2 = seg_A, seg_A
                a_idx_1, a_idx_2 = idx_A, idx_A
                req_keys = ['A']
        
                if state[key_1_index] == 1 and state[key_2_index] == 1:
                    if state[0] >= 0.9:
                        action = 6
                    else:
                        action = 1
                elif state[key_1_index] == 1 and state[key_2_index] == 0:
                    if state[0] < seg_2[0]:
                        action = 1
                    elif state[0] <= seg_2[1]:
                        action = a_idx_2
                    else:
                        action = 0                
                elif state[key_1_index] == 0 and state[key_2_index] == 1:
                    if state[0] < seg_1[0]:
                        action = 1
                    elif state[0] <= seg_1[1]:
                        action = a_idx_1
                    else:
                        action = 0           
                else:
                    if state[0] < seg_1[0]:
                        action = 1
                    elif state[0] <= seg_1[1]:
                        action = a_idx_1
                    elif state[0] < seg_2[0]:
                        action = 0
                    elif state[0] <= seg_2[1]:
                        action = a_idx_2
                    else:
                        action = 0                                   
                example['a'] = action
            
                if action not in masked_actions:
                    generated = True
        
            examples.append(example)
    
        dump_json(examples, f'embedding_network/bc_data/{name}_data.json')   

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    generate_data()