import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_train_expert import get_config
from bc_dataset import BehavioralCloningDataset
import numpy as np
import torch
import torch.optim as optim
from models import PolicyNetwork
import datetime
import os
import logging
import sys
from tqdm import tqdm
from copy import deepcopy
from utils import parse_env_spec, rollout

def performance(policy, env, eval_tasks, config):     
    """
    Evaluate the policy on the evaluation tasks.
    :param policy: torch.nn.Module
    :param env: gym.Env
    :param eval_tasks: list[numpy.ndarray(float)]
    :param config: argparse.ArgumentParser
    :returns: numpy.ndarray(float)
    """   
    policy.eval()
          
    perf = 0
    for s in eval_tasks:
        r = 0
        for _ in range(config.num_rollouts_per_task):
            r += int(rollout(policy, s, env, config.max_episode_len))
        perf += r / config.num_rollouts_per_task  
    perf /= len(eval_tasks)    
    
    return perf

def get_dataloader(input_path, batch_size):
    """
    Returns the training dataloader.
    :param input_path: str
    :param batch: int
    :returns: torch.utils.data.DataLoader
    """  
    dataset = BehavioralCloningDataset(f'{os.path.split(os.path.realpath(__file__))[0]}/{input_path}')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    
    return train_loader

def train_rl(policy, env, eval_tasks, config):
    """
    Trains the policy network using REINFORCE.
    :param policy: torch.nn.Module
    :param env: gym.Env
    :param eval_tasks: list[numpy.ndarray(float)]
    :param config: argparse.ArgumentParser
    """ 
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    max_perf = -np.inf
    max_perf_policy = None
    for i in tqdm(range(config.num_episodes)):
        state, ep_reward = env.reset(), 0
        for _ in range(config.max_episode_len):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + R
            returns.insert(0, R)  
        returns = torch.tensor(returns).to(config.device)
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        del policy.rewards[:]
        del policy.saved_log_probs[:]
        
        if i % config.log_interval == 0:
            perf = performance(deepcopy(policy), env, eval_tasks, config)
            logging.info('Episode {}\tLast Reward: {:.2f}\tPerformance: {:.4f}'.format(
                  i, ep_reward, perf))
            print('Episode {}\tLast Reward: {:.2f}\tPerformance: {:.4f}'.format(
                  i, ep_reward, perf))

            if perf > max_perf:
                max_perf = perf
                max_perf_policy = deepcopy(policy)
                logging.info('Max performance so far: {:.4f}'.format(perf))
                print('Max performance so far: {:.4f}'.format(perf))
    
    model_data = {
        'parameters': max_perf_policy.state_dict(),
        'performance': max_perf, 
        'model': "PolicyNetwork"
        }

    torch.save(model_data, os.path.join(config.output_path, "expert_policy.pt"))
        
    logging.info("Finished Training Expert")
    
def train_epoch_il(policy, epoch_idx, training_dataloader, optimizer, criterion, config):
    """
    Trains the policy network using behavioral cloning for one epoch.
    :param policy: torch.nn.Module
    :param epoch_idx: int
    :param training_dataloader: torch.utils.data.DataLoader
    :param optimizer: torch.optim.Optimizer
    :param criterion: torch.nn.NLLLoss
    :param config: argparse.ArgumentParser
    :returns: float
    """ 
    policy.train()

    loss_values = []

    for _, batch in enumerate(training_dataloader):
        S, A = batch
        S, A = S.to(config.device), A.to(config.device)
        
        A_pred = policy(S) + 1e-15
        
        loss = criterion(torch.log2(A_pred), A)

        optimizer.zero_grad()
        assert not torch.isnan(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        optimizer.step()

        loss_values.append(loss.item())
        
        break
    
    return np.mean(loss_values)    
    
def train_il(policy, env, eval_tasks, config):
    """
    Trains the policy network using behavioral cloning.
    :param policy: torch.nn.Module
    :param env: gym.Env
    :param eval_tasks: list[numpy.ndarray(float)]
    :param config: argparse.ArgumentParser
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr)
    train_loader = get_dataloader(config.input_path, config.batch_size)
    criterion = torch.nn.NLLLoss()
    
    max_perf = -np.inf
    max_perf_policy = None
    for epoch_idx in tqdm(range(1, config.num_epochs + 1)):
        mean_loss = train_epoch_il(policy, epoch_idx, train_loader, optimizer, criterion, config)
        
        if epoch_idx % config.log_interval == 0:
            perf = performance(deepcopy(policy), env, eval_tasks, config)
            logging.info('Epoch {}\tLoss: {:.2f}\tPerformance: {:.4f}'.format(
                  epoch_idx, mean_loss, perf))
            print('Episode {}\tLoss: {:.2f}\tPerformance: {:.4f}'.format(
                  epoch_idx, mean_loss, perf))

            if perf > max_perf:
                max_perf = perf
                max_perf_policy = deepcopy(policy)
                logging.info('Max performance so far: {:.4f}'.format(perf))
                print('Max performance so far: {:.4f}'.format(perf))
      
    model_data = {
        'parameters': max_perf_policy.state_dict(),
        'performance': max_perf, 
        'model': "PolicyNetwork"
        }

    torch.save(model_data, os.path.join(config.output_path, "expert_policy.pt"))
        
    logging.info("Finished Training Expert")        

if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)

    config = get_config()
    
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_train_expert", config.prefix)
    os.makedirs(os.path.join(config.output_path))

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Started Training Expert")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")
    
    env_name, env, eval_tasks = parse_env_spec(config.env_spec_path)
    
    np.random.seed(0)
    policy = PolicyNetwork(env_name, config.device).to(config.device)
    
    if config.pretrained_path is not None:
        model_data = torch.load(config.pretrained_path)
        policy = PolicyNetwork(env_name, config.device).to(config.device)
        policy.load_state_dict(model_data['parameters'])
    
    if config.technique == 'rl':
        train_rl(policy, env, eval_tasks, config)
    elif config.technique == 'il':
        train_il(policy, env, eval_tasks, config)
