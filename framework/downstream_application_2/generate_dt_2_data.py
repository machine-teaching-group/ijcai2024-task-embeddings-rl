import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_generate_dt_2_data import get_config
from multiprocessing import Pool
import numpy as np
import torch
from models import PolicyNetwork
from estimate_MI import estimate_MI
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

def get_result_query_1(MI_values):
    """
    Computes the ground truth answer to query 1.
    :param MI_values: list[float]
    :returns: int
    """ 
    return MI_values.index(max(MI_values)) + 1

def get_result_query_2_3(s_query, s_options, policies, env, config, MI_values, use_env_fn=False):
    """
    Computes the ground truth answers to query 2 and 3.
    :param s_query: numpy.ndarray(float)
    :param s_options: list[numpy.ndarray(float)]
    :param policies: list[torch.nn.Module]
    :param env: gym.Env
    :param config: argparse.ArgumentParser
    :param MI_values: list[float]
    :returns: int
    """ 
    s_options_harder = []
    s_options_easier = []
    p_success_s_query = get_p_success_task(s_query, policies, env, config, use_env_fn)
    for i, s in enumerate(s_options):
        p_success_s = get_p_success_task(s, policies, env, config, use_env_fn)
        if p_success_s + 0.01 < p_success_s_query:
            s_options_harder.append([s, i])
        elif p_success_s > p_success_s_query + 0.01:
            s_options_easier.append([s, i]) 
        
    MI_values_harder = [MI_values[s[1]] for s in s_options_harder]
    MI_values_easier = [MI_values[s[1]] for s in s_options_easier]
    
    if not s_options_harder:
        ans_2 = len(s_options) + 1
    else:
        ans_2 = s_options_harder[MI_values_harder.index(max(MI_values_harder))][1] + 1  
    
    if not s_options_easier:
        ans_3 = len(s_options) + 1  
    else:
        ans_3 = s_options_easier[MI_values_easier.index(max(MI_values_easier))][1] + 1
    
    return ans_2, ans_3

def gen_example(i):
    """
    Generates an example for the benchmark.
    :returns: dict
    """ 
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
    s_query = env_copy.reset()
        
    example['s_query'] = {'s': s_query.tolist(),
                             'expert_trajectory': get_trajectory(expert_policy, s_query, env_copy, config.max_episode_len, env_name == 'ICLR18')}
        
    options_data = {}
    results = []
    s_options = []
    for i in range(config.num_options):
        s_option_i = env_copy.reset()
        s_options.append(s_option_i)
        options_data[f's_option_{i}'] = {'s': s_option_i.tolist(),
                                         'expert_trajectory': get_trajectory(expert_policy, s_option_i, env_copy, config.max_episode_len, env_name == 'ICLR18')}       
    example['options_data'] = options_data
    
    MI_values = [estimate_MI(s_query, s, env, policies, config.max_episode_len, config.num_samples_1, env_name == 'ICLR18') for s in s_options]
        
    results.append(get_result_query_1(MI_values))
    ans_2, ans_3 = get_result_query_2_3(s_query, s_options, policies, env, config, MI_values, env_name == 'ICLR18')
    results.append(ans_2)
    results.append(ans_3)
    example['results'] = results
        
    return example

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
    
    pool = Pool()
    
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
         
    dump_json(data_train, '{}/data_train.json'.format(config.output_path))
    dump_json(data_test, '{}/data_test.json'.format(config.output_path))
    
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_generate_dt_2_data", config.prefix, str(config.num_options))
    os.makedirs(config.output_path)
    
    env_name, env, _ = parse_env_spec(config.env_spec_path)
    
    gen_data()
