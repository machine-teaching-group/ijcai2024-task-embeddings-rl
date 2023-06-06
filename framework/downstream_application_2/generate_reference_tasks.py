import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_generate_dt_2_data import get_config
import numpy as np
import torch
from models import PolicyNetwork
import os
from glob import glob
from tqdm import tqdm
import random
from copy import deepcopy
from utils import dump_json, parse_env_spec, rollout, get_trajectory

policies = None
expert_policy = None
env = None
env_name = None
config = None

def get_p_success_task(s, policies, env, config, use_env_fn=False):
    """
    Returns the probability of success of a random agent on a specific task.
    :param s: numpy.ndarray(float)
    :param policies: list[torch.nn.Module]
    :param env: gym.Env
    :param config: argparse.ArgumentParser
    :returns: float
    """
    n_success, n = 0, 0
    for policy in policies:
        for _ in range(config.num_samples_2):
            if rollout(policy, s, env, config.max_episode_len, use_env_fn):
                n_success += 1
            n += 1

    return n_success / n

def get_reference_tasks(policies, env, config, use_env_fn=False):
    """
    Returns a list of easy and hard reference tasks.
    :param policies: list[torch.nn.Module]
    :param env: gym.Env
    :param config: argparse.ArgumentParser
    :returns:
        - list[numpy.ndarray(float)]
        - list[numpy.ndarray(float)]
    """ 
    s_pool = [env.reset() for _ in range(config.pool_size)]
    s_pool_perfs = [get_p_success_task(s, policies, env, config, use_env_fn) for s in tqdm(s_pool)]
    
    indices = sorted(range(len(s_pool_perfs)), key=lambda i: s_pool_perfs[i])
    indices_easy = indices[-config.num_reference_tasks:]
    indices_hard = indices[:config.num_reference_tasks]
    
    return [s_pool[i] for i in indices_easy], [s_pool[i] for i in indices_hard]    

def gen_data():
    """
    Generates the benchmark.
    """ 
    global policies
    global expert_policy
    global env
    global env_name
    global config
    
    policies = []
    policies_index = []
    if env_name == 'ICLR18':
        for file in glob(f'embedding_network/{config.policies_path}' + f'/{config.prefix}' + '/*/models/*.model'):
            policies_index.append({'policy_path': file})
            model_path = file
            policy = torch.load(model_path, map_location=lambda storage, loc: storage)
            policy.eval()
            policies.append(policy) 

        expert_policy = None
    else:
        for file in glob(f'embedding_network/{config.policies_path}' + f'/{config.prefix}' + '/*/models/*.pt'):
            policies_index.append({'policy_path': file})
            model_data = torch.load(file)
            policy = PolicyNetwork(env_name, config.device).to(config.device)
            policy.load_state_dict(model_data['parameters'])
            policy.eval()
            policies.append(policy)  
        
        expert_policy_data = torch.load(f'embedding_network/{config.expert_policy_path}/{config.prefix}/expert_policy.pt') 
        expert_policy = PolicyNetwork(env_name, config.device).to(config.device)
        expert_policy.load_state_dict(expert_policy_data['parameters'])
        expert_policy.eval()
    
    easy_tasks = []
    hard_tasks = []
    s_e, s_h = get_reference_tasks(policies, env, config, env_name == 'ICLR18')
    for s in s_e:
        easy_tasks.append({'s': s.tolist(),
                           'expert_trajectory': get_trajectory(expert_policy, s, env, config.max_episode_len, env_name == 'ICLR18')})
    for s in s_h:
        hard_tasks.append({'s': s.tolist(),
                           'expert_trajectory': get_trajectory(expert_policy, s, env, config.max_episode_len, env_name == 'ICLR18')})  
    reference_tasks = {'easy': easy_tasks, 'hard': hard_tasks}         
        
    dump_json(reference_tasks, '{}/reference_tasks.json'.format(config.output_path))
    
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_generate_dt_2_data", config.prefix)
    
    env_name, env, _ = parse_env_spec(config.env_spec_path)
    
    gen_data()
