import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_dt_2_results_opt import get_config
from multiprocessing import Pool
import numpy as np
import torch
from models import PolicyNetwork
from estimate_MI import estimate_MI
import os
import logging
from glob import glob
from tqdm import tqdm
import sys
import random
from copy import deepcopy
from utils import load_json, parse_env_spec, rollout

dataset = None
policies = None
env = None
config = None

def get_p_success_task(s, policies, env, config):
    n_success, n = 0, 0
    for policy in policies:
        for _ in range(config.num_samples_2):
            if rollout(policy, s, env, config.max_episode_len):
                n_success += 1
            n += 1

    return n_success / n 

def get_result_query_1(config, MI_values):
    return sorted(range(1, len(MI_values) + 1), key=lambda j: -MI_values[j-1])[:config.k]

def get_result_query_2_3(s_query, s_options, policies, env, config, MI_values):
    s_options_harder = []
    s_options_easier = []
    p_success_s_query = get_p_success_task(s_query, policies, env, config)
    for j, s in enumerate(s_options):
        p_success_s = get_p_success_task(s, policies, env, config)
        if p_success_s + 0.01 < p_success_s_query:
            s_options_harder.append([s, j])
        elif p_success_s > p_success_s_query + 0.01:
            s_options_easier.append([s, j]) 
        
    MI_values_harder = [MI_values[s[1]] for s in s_options_harder]
    MI_values_easier = [MI_values[s[1]] for s in s_options_easier]
    
    if not s_options_harder:
        ans_2 = [len(s_options) + 1]
    else:
        sorted_harder_indices = sorted(range(len(MI_values_harder)), key=lambda j: -MI_values_harder[j])[:config.k]
        ans_2 = [s_options_harder[j][1] + 1 for j in sorted_harder_indices]
    
    if not s_options_easier:
        ans_3 = [len(s_options) + 1]  
    else:
        sorted_easier_indices = sorted(range(len(MI_values_easier)), key=lambda j: -MI_values_easier[j])[:config.k]
        ans_3 = [s_options_easier[j][1] + 1 for j in sorted_easier_indices]
    
    return ans_2, ans_3

def predict_example(i):
    global policies
    global env
    global config
    global dataset
    
    env_copy = deepcopy(env)
    
    torch.manual_seed(i+100)
    np.random.seed(i+100)
    random.seed(i+100)
    env_copy.observation_space.seed(i+100)

    example = dataset[i]
    s_query = np.asarray(example['s_query']['s'])
        
    s_options = []
    for option in example['options_data']:
        s_options.append(np.asarray(example['options_data'][option]['s']))
    
    MI_values = [estimate_MI(s_query, s, env, policies, config.max_episode_len, config.num_samples_1) for s in s_options]
        
    results = []    
    results.append(get_result_query_1(config, MI_values))
    ans_2, ans_3 = get_result_query_2_3(s_query, s_options, policies, env, config, MI_values)
    results.append(ans_2)
    results.append(ans_3)
        
    return [i, results]

def evaluate_technique(pred_test_outcomes, data_test):
    global config
    
    accuracies_list = []
    for k in range(1, config.k + 1):
        accuracies = {}
        for i in range(3):
            num_correct = 0
            for example, prediction in zip(data_test, pred_test_outcomes):
                if example['results'][i] in prediction[i][:k]:
                    num_correct += 1
            accuracies[f'Type_{i+1}'] = num_correct / len(data_test)  
        accuracies_list.append(accuracies) 
    
    return accuracies_list          
    
def run_techniques(env_name):
    global policies
    global env
    global config
    global dataset
    
    net_dir = os.path.split(os.path.realpath(__file__))[0]

    data_test = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.num_options}/data_test.json')[config.start_idx:config.start_idx + config.num_examples]
    
    dataset = data_test
    
    policies = []
    for file in glob(f'embedding_network/{config.policies_path}' + f'/{config.prefix}' + '/*/models/*.pt'):
        model_data = torch.load(file)
        policy = PolicyNetwork(env_name, config.device).to(config.device)
        policy.load_state_dict(model_data['parameters'])
        policy.eval()
        policies.append(policy)  
    
    random.shuffle(policies)
    policies = policies[:int(len(policies) * config.pop_frac)]    
    
    pool = Pool()
    
    total_examples = len(data_test)
    predictions = list(tqdm(pool.imap_unordered(predict_example,
                                                       range(total_examples)), total=total_examples))
    
    sorted_indices = sorted(range(len(predictions)), key=lambda i: predictions[i][0])
    sorted_predictions = [predictions[i][1] for i in sorted_indices]

    pred_accuracies_list = evaluate_technique(sorted_predictions, data_test)
    for k, pred_accuracies in zip(range(1, config.k + 1), pred_accuracies_list):
        logging.info('Top-{}: Our technique: Test Accuracies: {}'.format(k, pred_accuracies))
        print('Top-{}: Our technique: Test Accuracies: {}'.format(k, pred_accuracies))       
    
if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    
    torch.manual_seed(1000)
    np.random.seed(1000)
    random.seed(1000)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_dt_2_results", config.prefix, f'opt_{str(config.pop_frac)}', str(config.num_options))
    os.makedirs(config.output_path)
    
    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Results for Downstream Application 2")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")
    
    env_name, env, _ = parse_env_spec(config.env_spec_path)

    run_techniques(env_name)
