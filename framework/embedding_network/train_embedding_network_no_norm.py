import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from copy import deepcopy
from config_train_embedding_network import get_config
import numpy as np
import torch
import torch.utils.data
from models import EmbeddingNetwork
import os
import logging
import sys
from tqdm import tqdm
from utils import load_json, parse_env_spec

def dot_product_similarity(x1, x2):
    """
    Computes batch dot product of x1 and x2.
    :param x1: torch.Tensor
    :param x2: torch.Tensor
    :returns: torch.Tensor
    """ 
    return torch.bmm(x1.view(x1.size(0), 1, x1.size(1)), x2.view(x2.size(0), x2.size(1), 1))[:, 0, 0]

def get_dataloaders(env_name, config):
    """
    Returns dataloaders.
    :param env_name: str
    :param config: argparse.ArgumentParser
    :returns:
        - torch.utils.data.DataLoader
        - torch.utils.data.DataLoader
        - torch.utils.data.DataLoader
    """  
    net_dir = os.path.split(os.path.realpath(__file__))[0]
    
    if env_name == 'ICLR18':
        data_train = np.load(f'{net_dir}/{config.data_path}/{config.prefix}/MI_OrdinalConstraints/data_train_np.npy')
        data_val = np.load(f'{net_dir}/{config.data_path}/{config.prefix}/MI_OrdinalConstraints/data_val_np.npy')
        data_test = np.load(f'{net_dir}/{config.data_path}/{config.prefix}/MI_OrdinalConstraints/data_test_np.npy')

        data_train_reg = np.load(f'{net_dir}/{config.data_path}/{config.prefix}/Norm_OrdinalConstraints/data_train_np.npy')
        data_val_reg = np.load(f'{net_dir}/{config.data_path}/{config.prefix}/Norm_OrdinalConstraints/data_val_np.npy')
        data_test_reg = np.load(f'{net_dir}/{config.data_path}/{config.prefix}/Norm_OrdinalConstraints/data_test_np.npy')
    
        anchor_train = data_train[:, 0]
        anchor_val = data_val[:, 0]
        anchor_test = data_test[:, 0]
        positive_train = data_train[:, 1]
        positive_val = data_val[:, 1]
        positive_test = data_test[:, 1]
        negative_train = data_train[:, 2]
        negative_val = data_val[:, 2]  
        negative_test = data_test[:, 2]
        
        positive_reg_train = data_train_reg[:, 0]
        positive_reg_val = data_val_reg[:, 0]
        positive_reg_test = data_test_reg[:, 0]
        negative_reg_train = data_train_reg[:, 1]
        negative_reg_val = data_val_reg[:, 1]
        negative_reg_test = data_test_reg[:, 1]
    else:
        data_train = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/MI_OrdinalConstraints/data_train.json')
        data_val = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/MI_OrdinalConstraints/data_val.json')
        data_test = load_json(f'{net_dir}/{config.data_path}/{config.prefix}/MI_OrdinalConstraints/data_test.json')
    
        state_shape = np.asarray(data_train[0]['s_1']['s']).shape
        anchor_train = np.zeros((len(data_train), *(state_shape)))
        anchor_val = np.zeros((len(data_val), *(state_shape)))
        anchor_test = np.zeros((len(data_test), *(state_shape)))
        positive_train = np.zeros((len(data_train), *(state_shape)))
        positive_val = np.zeros((len(data_val), *(state_shape)))
        positive_test = np.zeros((len(data_test), *(state_shape)))
        negative_train = np.zeros((len(data_train), *(state_shape)))
        negative_val = np.zeros((len(data_val), *(state_shape)))
        negative_test = np.zeros((len(data_test), *(state_shape)))
    
        for i, example in enumerate(data_train):
            anchor_train[i] = np.asarray(example['s_1']['s'])
            positive_train[i] = np.asarray(example['s_2']['s'])
            negative_train[i] = np.asarray(example['s_3']['s'])
        
        for i, example in enumerate(data_val):
            anchor_val[i] = np.asarray(example['s_1']['s'])
            positive_val[i] = np.asarray(example['s_2']['s'])
            negative_val[i] = np.asarray(example['s_3']['s'])
        
        for i, example in enumerate(data_test):
            anchor_test[i] = np.asarray(example['s_1']['s'])
            positive_test[i] = np.asarray(example['s_2']['s'])
            negative_test[i] = np.asarray(example['s_3']['s'])     
    
        positive_reg_train = np.zeros((1000, *(state_shape)))
        positive_reg_val = np.zeros((1000, *(state_shape)))
        positive_reg_test = np.zeros((1000, *(state_shape)))
        negative_reg_train = np.zeros((1000, *(state_shape)))
        negative_reg_val = np.zeros((1000, *(state_shape)))
        negative_reg_test = np.zeros((1000, *(state_shape)))

    train_MI = torch.utils.data.TensorDataset(torch.Tensor(anchor_train), torch.Tensor(positive_train), torch.Tensor(negative_train))
    train_loader_MI = torch.utils.data.DataLoader(train_MI, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    
    val_MI = torch.utils.data.TensorDataset(torch.Tensor(anchor_val), torch.Tensor(positive_val), torch.Tensor(negative_val))
    val_loader_MI = torch.utils.data.DataLoader(val_MI, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    
    test_MI = torch.utils.data.TensorDataset(torch.Tensor(anchor_test), torch.Tensor(positive_test), torch.Tensor(negative_test))
    test_loader_MI = torch.utils.data.DataLoader(test_MI, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
   
    train_Norm = torch.utils.data.TensorDataset(torch.Tensor(positive_reg_train), torch.Tensor(negative_reg_train))
    train_loader_Norm = torch.utils.data.DataLoader(train_Norm, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    
    val_Norm = torch.utils.data.TensorDataset(torch.Tensor(positive_reg_val), torch.Tensor(negative_reg_val))
    val_loader_Norm = torch.utils.data.DataLoader(val_Norm, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)
    
    test_Norm = torch.utils.data.TensorDataset(torch.Tensor(positive_reg_test), torch.Tensor(negative_reg_test))
    test_loader_Norm = torch.utils.data.DataLoader(test_Norm, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=False)

    del anchor_train
    del anchor_val
    del anchor_test
    del positive_train
    del positive_val
    del positive_test
    del negative_train
    del negative_val
    del negative_test
    del positive_reg_train
    del positive_reg_val
    del positive_reg_test
    del negative_reg_train
    del negative_reg_val
    del negative_reg_test

    return train_loader_MI, train_loader_Norm, val_loader_MI, val_loader_Norm, test_loader_MI, test_loader_Norm

def train_epoch(model, epoch_idx, training_dataloader_MI, training_dataloader_Norm, optimizer, config):
    """
    Trains the embedding network for one epoch.
    :param model: torch.nn.Module
    :param epoch_idx: int
    :param training_dataloader_MI: torch.utils.data.DataLoader
    :param training_dataloader_Norm: torch.utils.data.DataLoader
    :param optimizer: torch.optim.Optimizer
    :param config: argparse.ArgumentParser
    :returns: float
    """ 
    model.train()

    loss_values = []

    dataloader_Norm_iterator = iter(training_dataloader_Norm)

    for _, batch in enumerate(training_dataloader_MI):
        anchor, positive, negative = batch
        
        try:
            positive_reg, negative_reg = next(dataloader_Norm_iterator)
        except StopIteration:
            dataloader_Norm_iterator = iter(training_dataloader_Norm)
            positive_reg, negative_reg = next(dataloader_Norm_iterator)

        anchor, positive, negative, positive_reg, negative_reg = anchor.to(config.device), positive.to(config.device), negative.to(config.device), positive_reg.to(config.device), negative_reg.to(config.device)    

        embedding_a = model(anchor)
        embedding_p = model(positive)
        embedding_n = model(negative)
        embedding_p_reg = model(positive_reg)
        embedding_n_reg = model(negative_reg)
        
        loss = torch.mean(torch.log2(1 + torch.exp(-(dot_product_similarity(embedding_a, embedding_p) - dot_product_similarity(embedding_a, embedding_n)))))
        loss += config.alpha * torch.mean(torch.log2(1 + torch.exp(-(torch.norm(embedding_n_reg, dim=1) - torch.norm(embedding_p_reg, dim=1)))))
        optimizer.zero_grad()
        assert not torch.isnan(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        loss_values.append(loss.item())
    
    mean_loss = np.mean(loss_values)

    if epoch_idx % config.log_interval == 0:
        logging.info("Epoch {0} Loss: {1}".format(epoch_idx, mean_loss))
        print("Epoch {0} Loss: {1}".format(epoch_idx, mean_loss))
        
    return mean_loss

def validate(model, validation_dataloader_MI, validation_dataloader_Norm, config):
    """
    Computes the loss on the validation set.
    :param model: torch.nn.Module
    :param validation_dataloader_MI: torch.utils.data.DataLoader
    :param validation_dataloader_Norm: torch.utils.data.DataLoader
    :param config: argparse.ArgumentParser
    :returns: float
    """ 
    model.eval()

    loss_values = []

    dataloader_Norm_iterator = iter(validation_dataloader_Norm)

    for _, batch in enumerate(validation_dataloader_MI):
        anchor, positive, negative = batch

        try:
            positive_reg, negative_reg = next(dataloader_Norm_iterator)
        except StopIteration:
            dataloader_Norm_iterator = iter(validation_dataloader_Norm)
            positive_reg, negative_reg = next(dataloader_Norm_iterator)

        anchor, positive, negative, positive_reg, negative_reg = anchor.to(config.device), positive.to(config.device), negative.to(config.device), positive_reg.to(config.device), negative_reg.to(config.device)
        
        embedding_a = model(anchor)
        embedding_p = model(positive)
        embedding_n = model(negative)
        embedding_p_reg = model(positive_reg)
        embedding_n_reg = model(negative_reg)
        
        loss = torch.mean(torch.log2(1 + torch.exp(-(dot_product_similarity(embedding_a, embedding_p) - dot_product_similarity(embedding_a, embedding_n)))))
        loss += config.alpha * torch.mean(torch.log2(1 + torch.exp(-(torch.norm(embedding_n_reg, dim=1) - torch.norm(embedding_p_reg, dim=1)))))
        loss_values.append(loss.item())

    mean_loss = np.mean(loss_values)
    logging.info("Validation Loss: {}".format(mean_loss))
    print("Validation Loss: {}".format(mean_loss))
    
    return mean_loss
    
def test(model, testing_dataloader_MI, testing_dataloader_Norm, config):
    """
    Computes the loss on the test set.
    :param model: torch.nn.Module
    :param testing_dataloader_MI: torch.utils.data.DataLoader
    :param testing_dataloader_Norm: torch.utils.data.DataLoader
    :param config: argparse.ArgumentParser
    :returns: float
    """ 
    model.eval()

    loss_values = []

    dataloader_Norm_iterator = iter(testing_dataloader_Norm)

    for _, batch in enumerate(testing_dataloader_MI):
        anchor, positive, negative = batch
        
        try:
            positive_reg, negative_reg = next(dataloader_Norm_iterator)
        except StopIteration:
            dataloader_Norm_iterator = iter(testing_dataloader_Norm)
            positive_reg, negative_reg = next(dataloader_Norm_iterator)

        anchor, positive, negative, positive_reg, negative_reg = anchor.to(config.device), positive.to(config.device), negative.to(config.device), positive_reg.to(config.device), negative_reg.to(config.device)
        
        embedding_a = model(anchor)
        embedding_p = model(positive)
        embedding_n = model(negative)
        embedding_p_reg = model(positive_reg)
        embedding_n_reg = model(negative_reg)
        
        loss = torch.mean(torch.log2(1 + torch.exp(-(dot_product_similarity(embedding_a, embedding_p) - dot_product_similarity(embedding_a, embedding_n)))))
        loss += config.alpha * torch.mean(torch.log2(1 + torch.exp(-(torch.norm(embedding_n_reg, dim=1) - torch.norm(embedding_p_reg, dim=1)))))
        loss_values.append(loss.item())

    mean_loss = np.mean(loss_values)
    logging.info("Testing Loss: {}".format(mean_loss))
    print("Testing Loss: {}".format(mean_loss))

def train(env_name, model, config):
    """
    Trains the embedding network.
    :param env_name: str
    :param model: torch.nn.Module
    :param config: argparse.ArgumentParser
    """ 
    training_dataloader_MI, training_dataloader_Norm, validation_dataloader_MI, validation_dataloader_Norm, testing_dataloader_MI, testing_dataloader_Norm \
        = get_dataloaders(env_name, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    min_val_loss = np.inf
    min_val_model = None
    for epoch_idx in tqdm(range(1, config.num_epochs + 1)):
        mean_loss = train_epoch(model, epoch_idx, training_dataloader_MI, training_dataloader_Norm, optimizer, config)
        if epoch_idx % config.val_interval == 0:
            val_loss = validate(model, validation_dataloader_MI, validation_dataloader_Norm, config)
            if val_loss < min_val_loss:
                logging.info("Minimum validation loss so far: {}".format(val_loss))
                print("Minimum validation loss so far: {}".format(val_loss))
                model.eval()
                min_val_model = deepcopy(model)
                min_val_loss = val_loss
    

    model_data = {
        'parameters': min_val_model.state_dict(),
        'model': "EmbeddingNetwork",
        'alpha': config.alpha
        }

    torch.save(model_data, '{}/embedding_model.pt'.format(config.output_path))
    
    logging.info("Finished Training Embedding Network")
    
    test(min_val_model, testing_dataloader_MI, testing_dataloader_Norm, config)

if __name__ == "__main__":
    run_id = np.random.randint(10000, 99999)
    
    config = get_config()

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    if config.output_path == "":
        config.output_path = os.path.split(os.path.realpath(__file__))[0]
    config.output_path = os.path.join(config.output_path, "runs_train_embedding", config.prefix, f"dim_{config.embedding_dim}")
    os.makedirs(config.output_path)

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Training Embedding Network")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")

    env_name, _, _ = parse_env_spec(config.env_spec_path)

    model = EmbeddingNetwork(env_name, config.embedding_dim).to(config.device)
    train(env_name, model, config)

