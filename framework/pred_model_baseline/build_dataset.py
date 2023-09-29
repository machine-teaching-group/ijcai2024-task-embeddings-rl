import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_build_dataset import get_config
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from utils import load_json, parse_env_spec

def build_dataset(config):
    data_path = os.path.split(os.path.realpath(__file__))[0]
    data_path = os.path.join(data_path, f'data/{config.env_name}')
    
    trajectories = load_json(f'{data_path}/trajectories/raw_trajectories.json')
    s_0_len, state_len, action_len = len(trajectories[0]['s_0']), len(trajectories[0]['trajectory'][0]['s']), \
        1 if isinstance(trajectories[0]['trajectory'][0]['a'], int) else len(trajectories[0]['trajectory'][0]['a'])
    print("Shape Info.:", s_0_len, state_len, action_len)
    
    dataset_s_0 = []
    dataset_transition = []
    dataset_state_n = []
    dataset_reward = []
    
    for traj in tqdm(trajectories):
        trajectory = traj['trajectory']
        s_0 = traj['s_0']
        for t, transition in enumerate(trajectory):
            dataset_s_0.append(s_0)
            if isinstance(transition['a'], int):
                dataset_transition.append(transition['s'] + [transition['a']])
            else:
                dataset_transition.append(transition['s'] + transition['a'])
            dataset_state_n.append(transition['s_n']) 
            dataset_reward.append(transition['r'])
    
    dataset_s_0 = np.asarray(dataset_s_0)
    dataset_transition = np.asarray(dataset_transition)
    dataset_state_n = np.asarray(dataset_state_n)
    dataset_reward = np.asarray(dataset_reward)
    
    output_path = f'{data_path}/dataset'    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)      
    
    np.save(f'{output_path}/dataset_s_0.npy', dataset_s_0)
    np.save(f'{output_path}/dataset_transition.npy', dataset_transition)
    np.save(f'{output_path}/dataset_state_n.npy', dataset_state_n)
    np.save(f'{output_path}/dataset_reward.npy', dataset_reward)

if __name__ == "__main__":
    config = get_config()
    
    config.env_name, config.env, _ = parse_env_spec(config.env_spec_path, return_env_class=False)
    
    if config.env_name == 'MultiKeyNav':
        config.max_episode_len = 40
    elif config.env_name == 'CartPoleVar':
        config.max_episode_len = 200
    elif config.env_name == 'PointMass':
        config.max_episode_len = 100
    elif config.env_name == 'BasicKarel':
        config.max_episode_len = 20

    build_dataset(config)