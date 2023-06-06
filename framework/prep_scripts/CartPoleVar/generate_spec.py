import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.CartPoleVar import CartPoleVarEnv
import numpy as np
import random
from utils import dump_json

def main():
    env = CartPoleVarEnv()
    
    tasks = []
    for _  in range(1000):
        tasks.append({'state': env.reset().tolist()})
        data = {"env_name": "CartPoleVar",
                "performance_eval_tasks": tasks}    
    
    dump_json(data, 'specs/CartPoleVar_spec.json')
    
if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    main()