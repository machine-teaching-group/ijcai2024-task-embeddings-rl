import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.PointMass import PointMassEnv
import numpy as np
import random
from utils import dump_json

def main():
    # Default
    env = PointMassEnv()
    
    tasks = []
    for _  in range(100):
        tasks.append({'state': env.reset().tolist()})
        data = {"env_name": "PointMass",
                "performance_eval_tasks": tasks}    
    
    dump_json(data, 'specs/PointMass_spec.json')
    
    # Left
    env = PointMassEnv(left=True)
    
    tasks = []
    for _  in range(100):
        tasks.append({'state': env.reset().tolist()})
        data = {"env_name": "PointMass",
                "env_params": [True],
                "performance_eval_tasks": tasks}    
    
    dump_json(data, 'specs/PointMass_Left_spec.json')
    
    # Right
    env = PointMassEnv(left=False)
    
    tasks = []
    for _  in range(100):
        tasks.append({'state': env.reset().tolist()})
        data = {"env_name": "PointMass",
                "env_params": [False],
                "performance_eval_tasks": tasks}    
    
    dump_json(data, 'specs/PointMass_Right_spec.json')
    
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    main()