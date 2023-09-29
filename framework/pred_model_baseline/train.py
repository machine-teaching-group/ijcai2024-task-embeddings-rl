import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_train import get_config
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from models import InferenceNetwork, DynamicsNetwork
import logging
from tqdm import tqdm
from copy import deepcopy
from utils import load_json, parse_env_spec

def get_dataloaders(config):
    dataset_path = os.path.join(config.data_path, f'data/{config.env_name}/dataset')
    dataset_s_0 = np.load(f'{dataset_path}/dataset_s_0.npy')
    dataset_transition = np.load(f'{dataset_path}/dataset_transition.npy')
    dataset_state_n = np.load(f'{dataset_path}/dataset_state_n.npy')
    dataset_reward = np.load(f'{dataset_path}/dataset_reward.npy')
    len_data = dataset_s_0.shape[0]

    if config.env_name in ['MultiKeyNav', 'BasicKarel']:
        pass
    elif config.env_name in ['CartPoleVar', 'PointMass']:
        dataset_transition = (dataset_transition - dataset_transition.mean(axis=0)) / dataset_transition.std(axis=0)
        dataset_state_n = (dataset_state_n - dataset_state_n.mean(axis=0)) / dataset_state_n.std(axis=0)

    train_data = torch.utils.data.TensorDataset(torch.Tensor(dataset_s_0[:int(0.7 * len_data)]), torch.Tensor(dataset_transition[:int(0.7 * len_data)]), torch.Tensor(dataset_state_n[:int(0.7 * len_data)]), torch.Tensor(dataset_reward[:int(0.7 * len_data)]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    
    val_data = torch.utils.data.TensorDataset(torch.Tensor(dataset_s_0[int(0.7 * len_data):]), torch.Tensor(dataset_transition[int(0.7 * len_data):]), torch.Tensor(dataset_state_n[int(0.7 * len_data):]), torch.Tensor(dataset_reward[int(0.7 * len_data):]))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
            
    del dataset_s_0
    del dataset_transition
    del dataset_state_n
    del dataset_reward

    return train_loader, val_loader

def train_epoch(embedding_model, dynamics_model, training_dataloader, optimizer, config, epoch_idx):
    embedding_model.train()
    dynamics_model.train()

    loss_s_n_values = []
    loss_r_values = []
    loss_kl_values = []
    loss_values = []

    for _, batch in enumerate(training_dataloader):
        s_0, transition, state_n, reward = batch
        s_0, transition, state_n, reward = s_0.to(config.device), transition.to(config.device), state_n.to(config.device), reward.to(config.device)
        
        task_embedding, q_z = embedding_model(s_0)
        dyn_inp = torch.cat((task_embedding, transition), dim=1)
        s_n_pred, r_pred = dynamics_model(dyn_inp)
        
        loss_s_n = F.mse_loss(s_n_pred, state_n).mean()
        loss_r = F.mse_loss(r_pred[:, 0], reward[:]).mean()
        loss_kl = torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1.)).sum(-1).mean()
        
        loss = config.alpha_1 * loss_s_n + config.alpha_2 * loss_r + config.alpha_3 * loss_kl
        
        optimizer.zero_grad()
        assert not torch.isnan(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(embedding_model.parameters()) + list(dynamics_model.parameters()), 2.0)
        optimizer.step()
        
        loss_s_n_values.append(loss_s_n.item())
        loss_r_values.append(loss_r.item())
        loss_kl_values.append(loss_kl.item())
        loss_values.append(loss.item())
    
    mean_loss_s_n = np.mean(loss_s_n_values)
    mean_loss_r = np.mean(loss_r_values)
    mean_loss_kl = np.mean(loss_kl_values)
    mean_loss = np.mean(loss_values)

    if epoch_idx % config.log_interval == 0:
        logging.info("Epoch {0} Loss: {1} Loss_s_n: {2} Loss_r: {3} Loss_kl: {4}".format(epoch_idx, mean_loss, mean_loss_s_n, mean_loss_r, mean_loss_kl))
        print("Epoch {0} Loss: {1} Loss_s_n: {2} Loss_r: {3} Loss_kl: {4}".format(epoch_idx, mean_loss, mean_loss_s_n, mean_loss_r, mean_loss_kl))
        
    return mean_loss

def validate(embedding_model, dynamics_model, validation_dataloader, config):
    embedding_model.eval()
    dynamics_model.eval()

    loss_s_n_values = []
    loss_r_values = []
    loss_kl_values = []
    loss_values = []
    
    #printed = False

    for _, batch in enumerate(validation_dataloader):
        s_0, transition, state_n, reward = batch
        s_0, transition, state_n, reward = s_0.to(config.device), transition.to(config.device), state_n.to(config.device), reward.to(config.device)

        task_embedding, q_z = embedding_model(s_0)
        dyn_inp = torch.cat((task_embedding, transition), dim=1)
        s_n_pred, r_pred = dynamics_model(dyn_inp)
        
        '''
        if not printed:
            printed = True
            print('s+a:', transition[0])
            print('s_n:', state_n[0])
            print('s_n_pred:', s_n_pred[0])
        '''    
        
        loss_s_n = F.mse_loss(s_n_pred, state_n).mean()
        loss_r = F.mse_loss(r_pred[:, 0], reward[:]).mean()
        loss_kl = torch.distributions.kl_divergence(q_z, torch.distributions.Normal(0, 1.)).sum(-1).mean()
        
        loss = config.alpha_1 * loss_s_n + config.alpha_2 * loss_r + config.alpha_3 * loss_kl
        
        loss_s_n_values.append(loss_s_n.item())
        loss_r_values.append(loss_r.item())
        loss_kl_values.append(loss_kl.item())
        loss_values.append(loss.item())
        
    mean_loss_s_n = np.mean(loss_s_n_values)
    mean_loss_r = np.mean(loss_r_values)
    mean_loss_kl = np.mean(loss_kl_values)
    mean_loss = np.mean(loss_values)

    logging.info("Validation Loss: {0} Loss_s_n: {1} Loss_r: {2} Loss_kl: {3}".format(mean_loss, mean_loss_s_n, mean_loss_r, mean_loss_kl))
    print("Validation Loss: {0} Loss_s_n: {1} Loss_r: {2} Loss_kl: {3}".format(mean_loss, mean_loss_s_n, mean_loss_r, mean_loss_kl))
        
    return mean_loss

def train(embedding_model, dynamics_model, config):
    training_dataloader, validation_dataloader = get_dataloaders(config)
    optimizer = torch.optim.Adam(list(embedding_model.parameters()) + list(dynamics_model.parameters()), lr=config.lr)
    
    min_val_loss = np.inf
    min_val_model = None
    for epoch_idx in tqdm(range(1, config.num_epochs + 1)):
        mean_loss = train_epoch(embedding_model, dynamics_model, training_dataloader, optimizer, config, epoch_idx)
        if epoch_idx % config.val_interval == 0:
            val_loss = validate(embedding_model, dynamics_model, validation_dataloader, config)
            if val_loss < min_val_loss:
                logging.info("Minimum validation loss so far: {}".format(val_loss))
                print("Minimum validation loss so far: {}".format(val_loss))
                embedding_model.eval()
                dynamics_model.eval()
                min_val_embedding_model = deepcopy(embedding_model)
                min_val_dynamics_model = deepcopy(dynamics_model)
                min_val_loss = val_loss
    

    embedding_model_data = {
        'parameters': min_val_embedding_model.state_dict(),
        'model': "InferenceNetwork",
        }
    
    dynamics_model_data = {
        'parameters': min_val_dynamics_model.state_dict(),
        'model': "DynamicsNetwork",
        }

    torch.save(embedding_model_data, f'{config.output_path}/embedding_model.pt')
    torch.save(dynamics_model_data, f'{config.output_path}/dynamics_model.pt')
    
    logging.info("Finished Training Embedding Network")

if __name__ == "__main__":
    config = get_config()

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    config.data_path = os.path.split(os.path.realpath(__file__))[0]
    config.env_name, _, _ = parse_env_spec(config.env_spec_path, return_env_class=False)
    
    config.output_path = f'{config.data_path}/networks/{config.env_name}/dim_{config.embedding_dim}'
    if not os.path.isdir(config.output_path):
        os.makedirs(config.output_path)

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log.txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Training Embedding Network")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    embedding_model = InferenceNetwork(config.env_name, config.embedding_dim).to(config.device)
    dynamics_model = DynamicsNetwork(config.env_name, config.embedding_dim).to(config.device)
    train(embedding_model, dynamics_model, config)