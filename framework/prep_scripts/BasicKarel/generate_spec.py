import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.BasicKarel import BasicKarelEnv
import numpy as np
import random
from glob import glob
from utils import load_json, dump_json

def main():
    env = BasicKarelEnv()
    
    tasks = []
    for file_path in glob('envs/data/basic_karel/data/val/task/*.json'):
        task_data = load_json(file_path) 
        tasks.append({'state': env.state_to_bitmaps(task_data).tolist()})
                                
    data = {"env_name": "BasicKarel",
    "performance_eval_tasks": tasks}    
    
    dump_json(data, 'specs/BasicKarel_spec.json')
    
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    main()