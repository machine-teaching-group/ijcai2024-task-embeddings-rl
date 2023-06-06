import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent_dir)

from envs.CartPoleVar import CartPoleVarEnv, CartPoleOriginalEnv
import numpy as np
import random
from tqdm import tqdm
from utils import dump_json 
from copy import deepcopy

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def generate_data():
    env = make_vec_env(CartPoleOriginalEnv, n_envs=4, seed=0)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1800000)
    model.save("prep_scripts/CartPoleVar/cartpole_expert")
    
    expert = PPO.load("prep_scripts/CartPoleVar/cartpole_expert")
    num_examples = 100000
    
    env = CartPoleVarEnv()
    
    for i in tqdm(range(5)):
        # (i == 0): Default
        # (i == 1): Bias: +ve Force and Type 0
        # (i == 2): Bias: +ve Force and Type 1
        # (i == 3): Bias: -ve Force and Type 0
        # (i == 4): Bias: -ve Force and Type 1
        
        gen_examples = 0
        examples = []       
        
        # 0: 0 +ve, 1: 0 -ve, 2: 1 +ve, 3: 1 -ve
        num_tasks = [None, None]
        num_tasks[0] = {0: 0, 1: 0, 2: 0, 3: 0}
        num_tasks[1] = {0: 0, 1: 0, 2: 0, 3: 0}
        while gen_examples < num_examples:
            state = env.reset()
            if i == 1:
                if not (state[env.key2idx["force_magnitude"]] > 0 and state[env.key2idx["task_type"]] == 0):
                    continue
            elif i == 2:
                if not (state[env.key2idx["force_magnitude"]] > 0 and state[env.key2idx["task_type"]] == 1):
                    continue  
            elif i == 3:
                if not (state[env.key2idx["force_magnitude"]] < 0 and state[env.key2idx["task_type"]] == 0):
                    continue  
            elif i == 4:
                if not (state[env.key2idx["force_magnitude"]] < 0 and state[env.key2idx["task_type"]] == 1):
                    continue           
            while gen_examples < num_examples:
                example = {}
                example['s'] = state.tolist()   
                task_type = state[env.key2idx["task_type"]]
                force_magnitude = state[env.key2idx["force_magnitude"]]
               
                state_temp = deepcopy(state)   
                state_temp[env.key2idx["task_type"]] = 0 
                state_temp[env.key2idx["force_magnitude"]] = abs(state[env.key2idx["force_magnitude"]])
                action, _ = expert.predict(state_temp, deterministic=True)
        
                if int(task_type) == 1:
                    action ^= 1
                if force_magnitude < 0:
                    action ^= 1    
                example['a'] = int(action)
                
                if i == 0:
                    examples.append(example)
                else:
                    example_1 = deepcopy(example)
                    example_2 = deepcopy(example)
                    example_2['s'][env.key2idx["force_magnitude"]] *= -1
                    example_3 = deepcopy(example)
                    example_3['s'][env.key2idx["task_type"]] = int(example_3['s'][env.key2idx["task_type"]]) ^ 1
                    example_4 = deepcopy(example)
                    example_4['s'][env.key2idx["force_magnitude"]] *= -1
                    example_4['s'][env.key2idx["task_type"]] = int(example_4['s'][env.key2idx["task_type"]]) ^ 1
                    examples.append(example_1)
                    examples.append(example_2)
                    examples.append(example_3)
                    examples.append(example_4)   
                gen_examples += 1
                

                state, reward, done, _ = env.step(action)
                if done:    
                    if force_magnitude >= 0:
                        if int(task_type) == 0:
                            num_tasks[int(reward)][0] += 1
                        else:
                            num_tasks[int(reward)][2] += 1 
                    else:
                        if int(task_type) == 0:
                            num_tasks[int(reward)][1] += 1
                        else:
                            num_tasks[int(reward)][3] += 1
                    break
                
        print(f'i: {i} Stats:', num_tasks)
        
        if i == 0:
            dump_json(examples, 'embedding_network/bc_data/CartPoleVar_default_data.json')
        elif i == 1:
            dump_json(examples, 'embedding_network/bc_data/CartPoleVar_bias_1_data.json')
        elif i == 2:
            dump_json(examples, 'embedding_network/bc_data/CartPoleVar_bias_2_data.json')
        elif i == 3:
            dump_json(examples, 'embedding_network/bc_data/CartPoleVar_bias_3_data.json')
        elif i == 4:
            dump_json(examples, 'embedding_network/bc_data/CartPoleVar_bias_4_data.json')        

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    generate_data()            