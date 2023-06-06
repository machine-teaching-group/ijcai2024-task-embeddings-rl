import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_dt_2_results import get_config
import numpy as np
import torch
from models import EmbeddingNetwork
import os
import logging
import sys
import random
from utils import load_json, parse_env_spec

def dot_product_similarity(x1, x2):
    return torch.bmm(x1.view(x1.size(0), 1, x1.size(1)), x2.view(x2.size(0), x2.size(1), 1))[:, 0, 0]

def get_embedding(task, embedding_model, device):
    with torch.no_grad():
        task_embedding = embedding_model(torch.Tensor(task).to(device).view(1, -1)).detach().cpu()

    return task_embedding

def get_norm(s, embedding_model, device):
    return torch.norm(get_embedding(s, embedding_model, device)).item()    

def get_similarity(s_1, s_2, embedding_model, device):
    e_1 = get_embedding(s_1, embedding_model, device)
    e_2 = get_embedding(s_2, embedding_model, device)
        
    return dot_product_similarity(e_1, e_2)[0]

def predict_result_query_1(s_query, s_options, embedding_model, device, config):
    similarities = [get_similarity(s_query, s, embedding_model, device) for s in s_options]

    return sorted(range(1, len(similarities) + 1), key=lambda i: -similarities[i-1])[:config.k]

def predict_result_query_2(s_query, s_options, embedding_model, device, config):
    s_options_harder = []
    for i, s in enumerate(s_options):
        if get_norm(s_query, embedding_model, device) \
                < get_norm(s, embedding_model, device):
                s_options_harder.append([s, i])
    
    if not s_options_harder:
        return [len(s_options) + 1]
    
    similarities = [get_similarity(s_query, s[0], embedding_model, device) for s in s_options_harder]
    
    sorted_sim_indices = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:config.k]
    
    return [s_options_harder[i][1] + 1 for i in sorted_sim_indices]

def predict_result_query_3(s_query, s_options, embedding_model, device, config):
    s_options_easier = []
    for i, s in enumerate(s_options):
        if get_norm(s_query, embedding_model, device) \
                > get_norm(s, embedding_model, device):
                s_options_easier.append([s, i])
    
    if not s_options_easier:
        return [len(s_options) + 1]
    
    similarities = [get_similarity(s_query, s[0], embedding_model, device) for s in s_options_easier]
    
    sorted_sim_indices = sorted(range(len(similarities)), key=lambda i: -similarities[i])[:config.k]
    
    return [s_options_easier[i][1] + 1 for i in sorted_sim_indices]

def evaluate_technique(pred_test_outcomes, data_test):
    accuracies = {}
    for i in range(3):
        num_correct = 0
        for example, prediction in zip(data_test, pred_test_outcomes):
            if example['results'][i] in prediction[i]:
                num_correct += 1
        accuracies[f'Type_{i+1}'] = num_correct / len(data_test)  
         
    return accuracies          
    
def run_techniques(env_name, config):
    net_dir = os.path.split(os.path.realpath(__file__))[0]
    
    data_train = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.num_options}/data_train.json')
    data_test = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.num_options}/data_test.json')[config.start_idx:config.start_idx + config.num_examples]
    
    # 1: Random Baseline
    predictions = []
    for example in data_test:
        prediction = []
        # Query 1
        l = list(range(1, len(data_test[0]['options_data'])))
        random.shuffle(l)
        prediction.append(l[:config.k])
        # Query 2
        l = list(range(1, len(data_test[0]['options_data'])))
        random.shuffle(l)
        prediction.append(l[:config.k])       
        # Query 3
        l = list(range(1, len(data_test[0]['options_data'])))
        random.shuffle(l)
        prediction.append(l[:config.k])
        
        predictions.append(prediction)
        
    pred_accuracies = evaluate_technique(predictions, data_test)
    logging.info('Random baseline: Test Accuracies: {}'.format(pred_accuracies))
    print('Random baseline: Test Accuracies: {}'.format(pred_accuracies)) 
    
    #---------
    
    model_data = torch.load(f'embedding_network/{config.embedding_net_path}/{config.prefix}/dim_{config.embedding_dim}/embedding_model.pt')
    embedding_model = EmbeddingNetwork(env_name, config.embedding_dim).to(config.device)
    embedding_model.load_state_dict(model_data['parameters'])
    embedding_model.eval() 
    
    # 1: Our technique
    predictions = []
    for example in data_test:
        s_query = example['s_query']['s']
        
        s_options = []
        for option in example['options_data']:
            s_options.append(example['options_data'][option]['s'])
        
        prediction = []
        # Query 1
        prediction.append(predict_result_query_1(s_query, s_options, embedding_model, config.device, config))
        # Query 2
        prediction.append(predict_result_query_2(s_query, s_options, embedding_model, config.device, config))
        # Query 3
        prediction.append(predict_result_query_3(s_query, s_options, embedding_model, config.device, config))
        
        predictions.append(prediction)

    pred_accuracies = evaluate_technique(predictions, data_test)
    logging.info('Our technique: Test Accuracies: {}'.format(pred_accuracies))
    print('Our technique: Test Accuracies: {}'.format(pred_accuracies))       
    
if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = get_config()

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_dt_2_results", config.prefix, 'our_approach', str(config.num_options))
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

    env_name, _, _ = parse_env_spec(config.env_spec_path)
    
    run_techniques(env_name, config)

