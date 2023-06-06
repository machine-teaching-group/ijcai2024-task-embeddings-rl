import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_dt_2_results import get_config
import numpy as np
import torch
import os
import logging
import sys
import random
from utils import load_json

def get_distance(s_1, s_2):
    f_1 = np.asarray(s_1)
    f_2 = np.asarray(s_2)
    
    return np.mean((f_1 - f_2)**2)

def get_avg_distance(s, S):
    avg_distance = np.inf
    for i in range(len(S)):
        avg_distance = min(avg_distance, get_distance(s, S[i]['s']))

    return avg_distance    

def predict_result_query_1(s_query, s_options, config):
    distances = [get_distance(s_query, s) for s in s_options]

    return sorted(range(1, len(distances) + 1), key=lambda i: distances[i-1])[:config.k]

def predict_result_query_2(s_query, s_options, reference_tasks, config):
    s_options_harder = []
    for i, s in enumerate(s_options):
        if get_avg_distance(s_query, reference_tasks['easy']) \
                < get_avg_distance(s, reference_tasks['easy']):
                s_options_harder.append([s, i])
    
    if not s_options_harder:
        return [len(s_options) + 1]
    
    distances = [get_distance(s_query, s[0]) for s in s_options_harder]
    
    sorted_dist_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:config.k]
    
    return [s_options_harder[i][1] + 1 for i in sorted_dist_indices]

def predict_result_query_3(s_query, s_options, reference_tasks, config):
    s_options_easier = []
    for i, s in enumerate(s_options):
        if get_avg_distance(s_query, reference_tasks['easy']) \
                > get_avg_distance(s, reference_tasks['easy']):
                s_options_easier.append([s, i])
    
    if not s_options_easier:
        return [len(s_options) + 1]
    
    distances = [get_distance(s_query, s[0]) for s in s_options_easier]
    
    sorted_dist_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:config.k]
    
    return [s_options_easier[i][1] + 1 for i in sorted_dist_indices]

def evaluate_technique(pred_test_outcomes, data_test):
    accuracies = {}
    for i in range(3):
        num_correct = 0
        for example, prediction in zip(data_test, pred_test_outcomes):
            if example['results'][i] in prediction[i]:
                num_correct += 1
        accuracies[f'Type_{i+1}'] = num_correct / len(data_test)  
         
    return accuracies          
    
def run_techniques(config):
    net_dir = os.path.split(os.path.realpath(__file__))[0]
    
    data_train = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.num_options}/data_train.json')
    data_test = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.num_options}/data_test.json')[config.start_idx:config.start_idx + config.num_examples]
    reference_tasks = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/reference_tasks.json')
    
    # 0: Visual Similarity
    predictions = []
    for example in data_test:
        s_query = example['s_query']['s']
        
        s_options = []
        for option in example['options_data']:
            s_options.append(example['options_data'][option]['s'])
        
        prediction = []
        # Query 1
        prediction.append(predict_result_query_1(s_query, s_options, config))
        # Query 2
        prediction.append(predict_result_query_2(s_query, s_options, reference_tasks, config))
        # Query 3
        prediction.append(predict_result_query_3(s_query, s_options, reference_tasks, config))
        
        predictions.append(prediction)

    pred_accuracies = evaluate_technique(predictions, data_test)
    logging.info('Visual Similarity: Test Accuracies: {}'.format(pred_accuracies))
    print('Visual Similarity: Test Accuracies: {}'.format(pred_accuracies))       
    
if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_dt_2_results", config.prefix, 'visual_similarity', str(config.num_options))
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
    
    run_techniques(config)

