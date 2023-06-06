import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_generate_dt_1_data import get_config
from multiprocessing import Pool
import numpy as np
import torch
from models import PolicyNetwork
import os
from glob import glob
from tqdm import tqdm
import random
from copy import deepcopy
from utils import dump_json, parse_env_spec, rollout, get_trajectory

p_success = None
policies = None
expert_policy = None
env = None
env_name = None
config = None

def get_p_success_agent(policy, env, config, use_env_fn=False):
    """
    Returns the probability of success of a specific agent on a random task.
    :param policy: torch.nn.Module
    :param env: gym.Env
    :param config: argparse.ArgumentParser
    :returns: float
    """ 
    n_success, n = 0, 0
    for _ in range(config.num_samples_1):
        s = env.reset()
        if rollout(policy, s, env, config.max_episode_len, use_env_fn):
            n_success += 1
        n += 1
        
    return n_success / n  

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

def get_p_success_agent_task(s, policy, env, config, use_env_fn=False):
    """
    Returns the probability of success of a specific agent on a specific task.
    :param s: numpy.ndarray(float)
    :param policy: torch.nn.Module
    :param env: gym.Env
    :param config: argparse.ArgumentParser
    :returns: float
    """ 
    n_success, n = 0, 0
    for _ in range(config.num_samples_3):
        if rollout(policy, s, env, config.max_episode_len, use_env_fn):
            n_success += 1
        n += 1
        
    return n_success / n  

def get_p_success(policies, env, config, use_env_fn=False):
    """
    Returns the probability of success of a random agent on a random task.
    :param policies: list[torch.nn.Module]
    :param env: gym.Env
    :param config: argparse.ArgumentParser
    :returns: float
    """
    n_success, n = 0, 0
    for _ in range(config.num_samples_4):
        s = env.reset()
        policy = random.choice(policies)
        if rollout(policy, s, env, config.max_episode_len, use_env_fn):
            n_success += 1
        n += 1
        
    return n_success / n 

def gen_example(i):
    """
    Generates an example for the benchmark.
    :returns: dict
    """ 
    global p_success
    global policies
    global expert_policy
    global env
    global env_name
    global config
    
    env_copy = env if env_name == 'ICLR18' else deepcopy(env)

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
    env_copy.observation_space.seed(i)
    
    example = {}
    s_test = env_copy.reset()
    policy_index = random.choice(list(range(len(policies))))
    policy = deepcopy(policies[policy_index])
        
    example['theta_index'] = policy_index
    example['s_test'] = {'s': s_test.tolist(), 
                         'o': int(rollout(policy, s_test, env_copy, config.max_episode_len, env_name == 'ICLR18')),
                         'expert_trajectory': get_trajectory(expert_policy, s_test, env_copy, config.max_episode_len, env_name == 'ICLR18')}
    example['p_success'] = p_success
    example['p_success_agent'] = get_p_success_agent(policy, env_copy, config, env_name == 'ICLR18')
    example['p_success_task'] = get_p_success_task(s_test, policies, env_copy, config, env_name == 'ICLR18')
    example['p_success_agent_task'] = get_p_success_agent_task(s_test, policy, env_copy, config, env_name == 'ICLR18')
        
    quiz_data = {}
    for i in range(config.quiz_size):
        s_quiz_i = env_copy.reset()
        quiz_data[f's_quiz_{i}'] = {'s': s_quiz_i.tolist(), 
                                    'o': int(rollout(policy, s_quiz_i, env_copy, config.max_episode_len, env_name == 'ICLR18')),
                                    'expert_trajectory': get_trajectory(expert_policy, s_quiz_i, env_copy, config.max_episode_len, env_name == 'ICLR18')}
    example['quiz_data'] = quiz_data
    
    return example

def gen_data():
    """
    Generates the benchmark.
    """ 
    global p_success
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
        
    p_success = get_p_success(policies, env, config, env_name == 'ICLR18')
    
    total_examples = config.train_size + config.test_size
    
    if env_name == 'ICLR18':
        generated_examples = []
        for i in tqdm(range(total_examples)):
            generated_examples.append(gen_example(i))
    else:
        pool = Pool()
        generated_examples = list(tqdm(pool.imap_unordered(gen_example,
                                                        range(total_examples)), total=total_examples))
    
    data_train = generated_examples[:config.train_size]
    data_test = generated_examples[config.train_size:]     
        
    dump_json(policies_index, '{}/policies_index.json'.format(config.output_path))    
    dump_json(data_train, '{}/data_train.json'.format(config.output_path))
    dump_json(data_test, '{}/data_test.json'.format(config.output_path))
    
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_generate_dt_1_data", config.prefix, str(config.quiz_size))
    os.makedirs(config.output_path)
    
    env_name, env, _ = parse_env_spec(config.env_spec_path)
    
    gen_data()
