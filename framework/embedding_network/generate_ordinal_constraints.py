import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_generate_ordinal_constraints import get_config
from multiprocessing import Pool
import numpy as np
import torch
from models import PolicyNetwork
from stable_baselines3 import SAC
from estimate_MI import estimate_MI
import random
import os
from glob import glob
from tqdm import tqdm
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

def gen_example_MI(i):
    """
    Generates a training example (ordinal constraint based on MI) for training the embedding network.
    :returns: dict
    """ 
    global policies
    global expert_policy
    global env
    global env_name
    global config

    env_copy = env if env_name in ['ICLR18'] else deepcopy(env)

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
    env_copy.observation_space.seed(i)

    generated = False
    
    while not generated:
        s_1 = env_copy.reset()
        s_2 = env_copy.reset()
        s_3 = env_copy.reset()
        
        expert_trajectory_s_1 = get_trajectory(expert_policy, s_1, env_copy, config.max_episode_len, env_name in ['ICLR18', 'PointMass'])
        expert_trajectory_s_2 = get_trajectory(expert_policy, s_2, env_copy, config.max_episode_len, env_name in ['ICLR18', 'PointMass'])
        expert_trajectory_s_3 = get_trajectory(expert_policy, s_3, env_copy, config.max_episode_len, env_name in ['ICLR18', 'PointMass'])
        
        I_hat_1_2 = estimate_MI(s_1, s_2, env_copy, policies, config.max_episode_len, config.num_samples_1, env_name in ['ICLR18', 'PointMass'])
        I_hat_1_3 = estimate_MI(s_1, s_3, env_copy, policies, config.max_episode_len, config.num_samples_1, env_name in ['ICLR18', 'PointMass'])

        if env_name in ['ICLR18']:
            if I_hat_1_2 > I_hat_1_3:
                example = {'s_1': {'s': s_1.tolist(), 'expert_trajectory': expert_trajectory_s_1},
                        's_2': {'s': s_2.tolist(), 'expert_trajectory': expert_trajectory_s_2},
                        's_3': {'s': s_3.tolist(), 'expert_trajectory': expert_trajectory_s_3},
                        'I_hat_1_2': I_hat_1_2,
                        'I_hat_1_3': I_hat_1_3}
                generated = True
            elif I_hat_1_3 > I_hat_1_2:
                example = {'s_1': {'s': s_1.tolist(), 'expert_trajectory': expert_trajectory_s_1}, 
                        's_2': {'s': s_3.tolist(), 'expert_trajectory': expert_trajectory_s_3},
                        's_3': {'s': s_2.tolist(), 'expert_trajectory': expert_trajectory_s_2},
                        'I_hat_1_2': I_hat_1_3,
                        'I_hat_1_3': I_hat_1_2}
                generated = True
        else:
            flag = np.random.binomial(1, 1 / (1 + np.exp(- 1000 * (I_hat_1_2 - I_hat_1_3))))
            if flag == 1:
                example = {'s_1': {'s': s_1.tolist(), 'expert_trajectory': expert_trajectory_s_1},
                        's_2': {'s': s_2.tolist(), 'expert_trajectory': expert_trajectory_s_2},
                        's_3': {'s': s_3.tolist(), 'expert_trajectory': expert_trajectory_s_3},
                        'I_hat_1_2': I_hat_1_2,
                        'I_hat_1_3': I_hat_1_3}
                generated = True
            else:
                example = {'s_1': {'s': s_1.tolist(), 'expert_trajectory': expert_trajectory_s_1}, 
                        's_2': {'s': s_3.tolist(), 'expert_trajectory': expert_trajectory_s_3},
                        's_3': {'s': s_2.tolist(), 'expert_trajectory': expert_trajectory_s_2},
                        'I_hat_1_2': I_hat_1_3,
                        'I_hat_1_3': I_hat_1_2}
                generated = True

    return example
    
def gen_example_Norm(i):
    """
    Generates a training example (ordinal constraint for embedding norms) for training the embedding network.
    :returns: dict
    """ 
    global policies
    global expert_policy
    global env
    global env_name
    global config

    env_copy = env if env_name in ['ICLR18'] else deepcopy(env)

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)
    env_copy.observation_space.seed(i)

    generated = False
    
    while not generated:
        s_1 = env_copy.reset()
        s_2 = env_copy.reset()
        
        expert_trajectory_s_1 = get_trajectory(expert_policy, s_1, env_copy, config.max_episode_len, env_name in ['ICLR18', 'PointMass'])
        expert_trajectory_s_2 = get_trajectory(expert_policy, s_2, env_copy, config.max_episode_len, env_name in ['ICLR18', 'PointMass'])
        
        pos_1 = get_p_success_task(s_1, policies, env_copy, config, env_name in ['ICLR18', 'PointMass'])
        pos_2 = get_p_success_task(s_2, policies, env_copy, config, env_name in ['ICLR18', 'PointMass'])
        
        if pos_1 > pos_2 + 0.1:
            example = {'s_1': {'s': s_1.tolist(), 'expert_trajectory': expert_trajectory_s_1},
                    's_2': {'s': s_2.tolist(), 'expert_trajectory': expert_trajectory_s_2},
                    'pos_1': pos_1,
                    'pos_2': pos_2}
            generated = True
        elif pos_2 > pos_1 + 0.1:
            example = {'s_1': {'s': s_2.tolist(), 'expert_trajectory': expert_trajectory_s_2}, 
                    's_2': {'s': s_1.tolist(), 'expert_trajectory': expert_trajectory_s_1},
                    'pos_1': pos_2,
                    'pos_2': pos_1}
            generated = True

    return example

def gen_data():
    """
    Generates the training data for the embedding network.
    """ 
    global policies
    global expert_policy
    global env
    global env_name
    global config
    
    net_dir = os.path.split(os.path.realpath(__file__))[0]
    
    policies = []
    if env_name == 'ICLR18':
        for file in glob(f'{net_dir}/{config.policies_path}' + f'/{config.prefix}' + '/*/models/*.model'):
            model_path = file
            policy = torch.load(model_path, map_location=lambda storage, loc: storage)
            policy.eval()
            policies.append(policy) 

        expert_policy = None
    elif env_name == 'PointMass':
        for file in glob(f'{net_dir}/{config.policies_path}' + f'/{config.prefix}' + '/*/models/*.pt'):
            model_path = file
            policy = SAC.load(model_path, env=env, device="cpu")
            policies.append(policy)
            
        expert_policy = None
    else:    
        for file in glob(f'{net_dir}/{config.policies_path}' + f'/{config.prefix}' + '/*/models/*.pt'):
            model_data = torch.load(file)
            policy = PolicyNetwork(env_name, config.device).to(config.device)
            policy.load_state_dict(model_data['parameters'])
            policy.eval()
            policies.append(policy)
    
        expert_policy_data = torch.load(f'{net_dir}/{config.expert_policy_path}/{config.prefix}/expert_policy.pt') 
        expert_policy = PolicyNetwork(env_name, config.device).to(config.device)
        expert_policy.load_state_dict(expert_policy_data['parameters'])
        expert_policy.eval()

    total_examples = config.train_size + config.val_size + config.test_size
    suffix = '_{}'.format(config.exp) if env_name == 'ICLR18' else ''

    if env_name in ['ICLR18']:
        generated_examples_MI = []
        for i in tqdm(range(total_examples)):
            generated_examples_MI.append(gen_example_MI(100000 * int(config.exp) + i))
    else:
        pool = Pool()
        generated_examples_MI = list(tqdm(pool.imap_unordered(gen_example_MI,
                                                        range(total_examples)), total=total_examples))
                                                       
    data_train_MI = generated_examples_MI[:config.train_size]
    data_val_MI = generated_examples_MI[config.train_size:config.train_size+config.val_size]
    data_test_MI = generated_examples_MI[config.train_size+config.val_size:]
    
    if not os.path.isdir('{}/MI_OrdinalConstraints'.format(config.output_path)):
        os.mkdir('{}/MI_OrdinalConstraints'.format(config.output_path))
    
    dump_json(data_train_MI, '{}/MI_OrdinalConstraints/data_train{}.json'.format(config.output_path, suffix))
    dump_json(data_val_MI, '{}/MI_OrdinalConstraints/data_val{}.json'.format(config.output_path, suffix))
    dump_json(data_test_MI, '{}/MI_OrdinalConstraints/data_test{}.json'.format(config.output_path, suffix))

    if env_name in ['ICLR18']:
        generated_examples_Norm = []
        for i in tqdm(range(total_examples)):
            generated_examples_Norm.append(gen_example_Norm(100000 * int(config.exp) + i))      
    else:                                               
        generated_examples_Norm = list(tqdm(pool.imap_unordered(gen_example_Norm,
                                                        range(total_examples)), total=total_examples))
                                                       
    data_train_Norm = generated_examples_Norm[:config.train_size]
    data_val_Norm = generated_examples_Norm[config.train_size:config.train_size+config.val_size]
    data_test_Norm = generated_examples_Norm[config.train_size+config.val_size:]
    
    if not os.path.isdir('{}/Norm_OrdinalConstraints'.format(config.output_path)):
        os.mkdir('{}/Norm_OrdinalConstraints'.format(config.output_path))
    
    dump_json(data_train_Norm, '{}/Norm_OrdinalConstraints/data_train{}.json'.format(config.output_path, suffix))
    dump_json(data_val_Norm, '{}/Norm_OrdinalConstraints/data_val{}.json'.format(config.output_path, suffix))
    dump_json(data_test_Norm, '{}/Norm_OrdinalConstraints/data_test{}.json'.format(config.output_path, suffix))

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_generate_MI_data", config.prefix)
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)

    env_name, env, _ = parse_env_spec(config.env_spec_path)

    gen_data()