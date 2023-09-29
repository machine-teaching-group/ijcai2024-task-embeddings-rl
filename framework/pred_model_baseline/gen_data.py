import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_gen_data import get_config
import numpy as np
import torch
from models import PolicyNetwork
import random
from tqdm import tqdm
from utils import dump_json, parse_env_spec, get_trajectory_trans

from stable_baselines3 import SAC

def gen_data(config):
    model_path = f'embedding_network/{config.expert_policy_path}/{config.prefix}/expert_policy.pt'
    
    if config.env_name == 'PointMass':
        expert_policy = SAC.load(model_path, env=config.env, device="cpu")
    else:
        model_data = torch.load(model_path)
        expert_policy = PolicyNetwork(config.env_name, config.device).to(config.device)
        expert_policy.load_state_dict(model_data['parameters'])
        expert_policy.eval()
    
    trajectories = []
    for _ in tqdm(range(config.num_trajectories)):
        while True:
            s_0, trajectory, solved = get_trajectory_trans(expert_policy, config.env_name, config.env, config.max_episode_len, config.one_hot_action, config.n_context_idx)
            if solved:
                break
        trajectories.append({'s_0': s_0, 'trajectory': trajectory})
    
    output_path = os.path.split(os.path.realpath(__file__))[0]
    output_path = os.path.join(output_path, f'data/{config.env_name}/trajectories')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)    
    dump_json(trajectories, f'{output_path}/raw_trajectories.json')        

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = get_config()
    
    config.env_name, config.env, _ = parse_env_spec(config.env_spec_path, return_env_class=False)
    
    if config.env_name == 'MultiKeyNav':
        config.max_episode_len = 40
        config.one_hot_action = 7
        config.n_context_idx = [0, 1, 2, 3, 4]
    elif config.env_name == 'CartPoleVar':
        config.max_episode_len = 200
        config.one_hot_action = None
        config.n_context_idx = [0, 1, 2, 3, 6]
    elif config.env_name == 'PointMass':
        config.max_episode_len = 100
        config.one_hot_action = None
        config.n_context_idx = [0, 1, 2, 3]
    elif config.env_name == 'BasicKarel':
        config.max_episode_len = 20
        config.one_hot_action = 6
        config.n_context_idx = list(range(0, 16)) + list(range(16, 32)) + list(range(48, 64)) + list(range(80, 84))

    gen_data(config)