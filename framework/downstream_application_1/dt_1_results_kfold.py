import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_dt_1_results import get_config
from sklearn.model_selection import KFold
import numpy as np
import torch
from models import EmbeddingNetwork, InferenceNetwork
import statistics
import os
import logging
import sys
import random
from utils import load_json, parse_env_spec

def get_embedding(task, embedding_model, device, pred_model=False):
    with torch.no_grad():
        if pred_model:
            task_embedding = embedding_model(torch.Tensor(task).to(device).view(1, -1))[0].detach().cpu().numpy()[0]
        else:
            task_embedding = embedding_model(torch.Tensor(task).to(device).view(1, -1)).detach().cpu().numpy()[0]

    return task_embedding

def get_soft_NN_prediction(task_embedding, quiz_embeddings, quiz_perf, beta):
    h = [np.exp(-beta * np.sum((q_e - task_embedding) ** 2)) for q_e in quiz_embeddings]
    sum_h = sum(h)
    h = [i / sum_h for i in h]
    return sum([h_q * O_q  for O_q, h_q in zip(quiz_perf, h)]) > 0.5

def evaluate_technique(pred_test_outcomes, data_test):
    num_correct = 0
    for example, prediction in zip(data_test, pred_test_outcomes):
        if example['s_test']['o'] == prediction:
            num_correct += 1
            
    return num_correct / len(data_test)        
    
def run_techniques(env_name, config):
    logging.info(f'Sub-Quiz Size: {config.sub_size}')
    print(f'Sub-Quiz Size: {config.sub_size}')

    net_dir = os.path.split(os.path.realpath(__file__))[0]
    
    data_train = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.quiz_size}/data_train.json')
    data_test = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/{config.quiz_size}/data_test.json')
    
    kf = KFold(n_splits=config.num_splits, shuffle=True)
    
    data_test_folds = []
    for _, test_index in kf.split(np.zeros(len(data_test))):
        data_test_sub = [data_test[i] for i in test_index]
        data_test_folds.append(data_test_sub)

    data_train_folds = []
    for _, train_index in kf.split(np.zeros(len(data_train))):
        data_train_sub = [data_train[i] for i in train_index]
        data_train_folds.append(data_train_sub)    
    
    # 0: Random Baseline
    pred_accuracies = []

    for data_test_sub in data_test_folds:
        predictions = [random.choice([0, 1]) for _ in range(len(data_test_sub))]
        pred_accuracy = evaluate_technique(predictions, data_test_sub)

        pred_accuracies.append(pred_accuracy)

    logging.info('Random: Test Accuracies: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    print('Random: Test Accuracies: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))

    # 1: AA-TA (Use p_success)
    pred_accuracies = []
    
    for data_test_sub in data_test_folds:
        p_success = data_test_sub[0]['p_success']
        if p_success >= 0.5:
            pred_accuracy = evaluate_technique([1] * len(data_test_sub), data_test_sub)
        else:
            pred_accuracy = evaluate_technique([0] * len(data_test_sub), data_test_sub)    
        
        pred_accuracies.append(pred_accuracy)    
        
    logging.info('AA-TA: Test Accuracies: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    print('AA-TA: Test Accuracies: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    
    # 2: IgnoreTask (Use p_success_agent)
    pred_accuracies = []
    
    for data_test_sub in data_test_folds:
        predictions = [1 if example['p_success_agent'] > 0.5 else 0 for example in data_test_sub]
        pred_accuracy = evaluate_technique(predictions, data_test_sub)
        
        pred_accuracies.append(pred_accuracy)
    
    logging.info('IgnoreTask: Test Accuracy: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    print('IgnoreTask: Test Accuracy: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    
    # 3: IgnoreAgent (Use p_success_task)
    pred_accuracies = []
    
    for data_test_sub in data_test_folds:
        predictions = [1 if example['p_success_task'] > 0.5 else 0 for example in data_test_sub]
        pred_accuracy = evaluate_technique(predictions, data_test_sub)
        
        pred_accuracies.append(pred_accuracy)
    
    logging.info('IgnoreAgent: Test Accuracy: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    print('IgnoreAgent: Test Accuracy: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    
    #4: OPT (Use p_success_agent_task)
    pred_accuracies = []
    
    for data_test_sub in data_test_folds:
        predictions = [1 if example['p_success_agent_task'] > 0.5 else 0 for example in data_test_sub]
        pred_accuracy = evaluate_technique(predictions, data_test_sub)
        
        pred_accuracies.append(pred_accuracy)
    
    logging.info('OPT: Test Accuracy: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    print('OPT: Test Accuracy: mean: {}, s.d.: {}'.format(statistics.mean(pred_accuracies), statistics.stdev(pred_accuracies)))
    
    #---------
    
    model_data = torch.load(f'pred_model_baseline/networks/{env_name}/dim_{config.embedding_dim_pred}/embedding_model.pt')
    embedding_model = InferenceNetwork(env_name, config.embedding_dim_pred).to(config.device)
    embedding_model.load_state_dict(model_data['parameters'])
    embedding_model.eval() 
    
    # 5: (Predictive Model) Soft-NN

    soft_nn_pred_accuracies = []
    
    t_e = []
    q_e = []
    q_p = []
    for example in data_train:
        task_embedding = get_embedding(example['s_test']['s'], embedding_model, config.device, pred_model=True)
        t_e.append(task_embedding)
        
        quiz_embeddings = []
        quiz_perf = []
        for i, key in enumerate(example['quiz_data']):
            if i >= config.sub_size:
                break

            quiz_embeddings.append(get_embedding(example['quiz_data'][key]['s'], embedding_model, config.device, pred_model=True))
            quiz_perf.append(example['quiz_data'][key]['o'])
        
        q_e.append(quiz_embeddings)
        q_p.append(quiz_perf)  
    
    best_beta = 0
    best_train_acc = -np.inf
    betas = [0.01 * i for i in range(0, 100)]
    for beta in betas:
        preds = []
        for task_embedding, quiz_embeddings, quiz_perf in zip(t_e, q_e, q_p):
            preds.append(get_soft_NN_prediction(task_embedding, quiz_embeddings, quiz_perf, beta=beta))
        train_acc = evaluate_technique(preds, data_train)
        if train_acc > best_train_acc:
            best_beta = beta
            best_train_acc = train_acc
    
    for data_test_sub in data_test_folds:
        soft_nn_predictions = []
    
        for example in data_test_sub:
            task_embedding = get_embedding(example['s_test']['s'], embedding_model, config.device, pred_model=True)
        
            quiz_embeddings = []
            quiz_perf = []
            for i, key in enumerate(example['quiz_data']):
                if i >= config.sub_size:
                    break

                quiz_embeddings.append(get_embedding(example['quiz_data'][key]['s'], embedding_model, config.device, pred_model=True))
                quiz_perf.append(example['quiz_data'][key]['o']) 
          
            soft_nn_predictions.append(get_soft_NN_prediction(task_embedding, quiz_embeddings, quiz_perf, beta=best_beta))
        
        pred_accuracy = evaluate_technique(soft_nn_predictions, data_test_sub)
        soft_nn_pred_accuracies.append(pred_accuracy)
    
    logging.info('(PredModel) Soft-NN (beta={}): Test Accuracy: mean: {}, s.d.: {}'.format(best_beta, statistics.mean(soft_nn_pred_accuracies), statistics.stdev(soft_nn_pred_accuracies)))
    print('(PredModel) Soft-NN (beta={}): Test Accuracy: mean: {}, s.d.: {}'.format(best_beta, statistics.mean(soft_nn_pred_accuracies), statistics.stdev(soft_nn_pred_accuracies)))
    
    #---------
    
    model_data = torch.load(f'embedding_network/{config.embedding_net_path}/{config.prefix}/dim_{config.embedding_dim}/embedding_model.pt')
    embedding_model = EmbeddingNetwork(env_name, config.embedding_dim).to(config.device)
    embedding_model.load_state_dict(model_data['parameters'])
    embedding_model.eval() 
    
    # 6: (Ours) Soft-NN

    soft_nn_pred_accuracies = []
    
    t_e = []
    q_e = []
    q_p = []
    for example in data_train:
        task_embedding = get_embedding(example['s_test']['s'], embedding_model, config.device)
        t_e.append(task_embedding)
        
        quiz_embeddings = []
        quiz_perf = []
        for i, key in enumerate(example['quiz_data']):
            if i >= config.sub_size:
                break

            quiz_embeddings.append(get_embedding(example['quiz_data'][key]['s'], embedding_model, config.device))
            quiz_perf.append(example['quiz_data'][key]['o'])
        
        q_e.append(quiz_embeddings)
        q_p.append(quiz_perf)  
    
    best_beta = 0
    best_train_acc = -np.inf
    betas = [0.01 * i for i in range(0, 100)]
    for beta in betas:
        preds = []
        for task_embedding, quiz_embeddings, quiz_perf in zip(t_e, q_e, q_p):
            preds.append(get_soft_NN_prediction(task_embedding, quiz_embeddings, quiz_perf, beta=beta))
        train_acc = evaluate_technique(preds, data_train)
        if train_acc > best_train_acc:
            best_beta = beta
            best_train_acc = train_acc
    
    for data_test_sub in data_test_folds:
        soft_nn_predictions = []
    
        for example in data_test_sub:
            task_embedding = get_embedding(example['s_test']['s'], embedding_model, config.device)
        
            quiz_embeddings = []
            quiz_perf = []
            for i, key in enumerate(example['quiz_data']):
                if i >= config.sub_size:
                    break

                quiz_embeddings.append(get_embedding(example['quiz_data'][key]['s'], embedding_model, config.device))
                quiz_perf.append(example['quiz_data'][key]['o']) 
          
            soft_nn_predictions.append(get_soft_NN_prediction(task_embedding, quiz_embeddings, quiz_perf, beta=best_beta))
        
        pred_accuracy = evaluate_technique(soft_nn_predictions, data_test_sub)
        soft_nn_pred_accuracies.append(pred_accuracy)
    
    logging.info('(Ours) Soft-NN (beta={}): Test Accuracy: mean: {}, s.d.: {}'.format(best_beta, statistics.mean(soft_nn_pred_accuracies), statistics.stdev(soft_nn_pred_accuracies)))
    print('(Ours) Soft-NN (beta={}): Test Accuracy: mean: {}, s.d.: {}'.format(best_beta, statistics.mean(soft_nn_pred_accuracies), statistics.stdev(soft_nn_pred_accuracies)))
    
if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)

    config = get_config()
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_dt_1_results", config.prefix, str(config.quiz_size))
    os.makedirs(config.output_path)
    
    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Results for Downstream Application 1")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    env_name, _, _ = parse_env_spec(config.env_spec_path)
    
    run_techniques(env_name, config)

