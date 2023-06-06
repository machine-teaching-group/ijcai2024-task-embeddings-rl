import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.MultiKeyNav import MultiKeyNavEnv
from envs.MultiKeyNavA import MultiKeyNavAEnv
from envs.MultiKeyNavAB import MultiKeyNavABEnv
import numpy as np
import random
from glob import glob
from utils import load_json, dump_json

def main():
    envs = [['MultiKeyNav', MultiKeyNavEnv()],
            ['MultiKeyNavA', MultiKeyNavAEnv()],
            ['MultiKeyNavAB', MultiKeyNavABEnv()]]
    
    for env_name, env in envs:
        tasks = []
        
        for loc in [0.05, 0.45, 0.85]:
            for key_A in [0, 1]:
                for key_B in [0, 1]:
                    for key_C in [0, 1]:
                        for key_D in [0, 1]:
                            for door_bit_1 in [0, 1]:
                                for door_bit_2 in [0, 1]:
                                    tasks.append({'state': [loc, key_A, key_B, key_C, key_D, door_bit_1, door_bit_2]})
                                
        data = {"env_name": env_name,
        "performance_eval_tasks": tasks}    
    
        dump_json(data, f'specs/{env_name}_spec.json')
    
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    main()