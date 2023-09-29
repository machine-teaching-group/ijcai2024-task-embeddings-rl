import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from envs.nps.network import IOsEncoder

class PolicyNetwork(nn.Module):
    def __init__(self, env_name, device, action_mask=None):
        super(PolicyNetwork, self).__init__()
        self.env_name = env_name
        
        if env_name in ['MultiKeyNav', 'MultiKeyNavA', 'MultiKeyNavAB']:
            self.fc1 = nn.Linear(7, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 7)
        elif env_name == 'BasicKarel':
            self.fc1 = nn.Linear(88, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 6)
        elif env_name == 'CartPoleVar':
            self.fc1 = nn.Linear(7, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 2)   

        self.saved_log_probs = []
        self.rewards = []
        
        self.device = device
        
        if action_mask is not None:
            self.action_mask = torch.Tensor(list(map(int, action_mask.split(" ")))).view(1, -1).to(device)
            self.action_mask[self.action_mask == 0] = - float('inf')
            self.action_mask[self.action_mask == 1] = 0
        else:
            self.action_mask = None

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_preferences = self.fc3(x)
        
        if self.action_mask is not None:
            action_preferences = action_preferences + self.action_mask

        return F.softmax(action_preferences, dim=1)

    def select_action(self, state, greedy=False):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        probs = self(state)
        m = Categorical(probs)

        if greedy:
            action = torch.argmax(probs)
        else:
            action = m.sample()
        self.saved_log_probs.append(m.log_prob(action.to(self.device)))

        return action.item() 

class EmbeddingNetwork(nn.Module):
    def __init__(self, env_name, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.env_name = env_name
        
        if env_name in ['MultiKeyNav', 'MultiKeyNavA', 'MultiKeyNavAB']:
            self.fc1 = nn.Linear(7, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, embedding_dim)
        elif env_name == 'BasicKarel':
            self.fc1 = nn.Linear(88, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, embedding_dim)
        elif env_name == 'CartPoleVar':
            self.fc1 = nn.Linear(7, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, embedding_dim)  
        elif env_name == 'ICLR18':
            self.io_encoder = torch.load("envs/data/karel/snapshot_weights_95.model").encoder.cpu() 
        
            self.fc_1 = nn.Linear(512, 256)
            self.fc_2 = nn.Linear(256, embedding_dim)
        elif env_name == 'PointMass':
            self.fc1 = nn.Linear(7, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, embedding_dim)
    
    def encoder(self, state):
        input_grids = state[:, 1 : 1 + 25920].view(state.size(0), 5, 16, 18, 18)
        output_grids = state[:, 1 + 25920:].view(state.size(0), 5, 16, 18, 18)
        io_embedding = self.io_encoder(input_grids, output_grids)

        return io_embedding            

    def forward(self, x):
        if self.env_name == 'ICLR18':
            # 5 * 16 * 18 * 18 = 25920
            io_embedding = self.encoder(x)
            io_embedding = torch.mean(io_embedding, dim=1)
            e = F.relu(self.fc_1(io_embedding))
            e = self.fc_2(e)

            return e
            
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
    
class InferenceNetwork(nn.Module):
    def __init__(self, env_name, embedding_dim):
        super(InferenceNetwork, self).__init__()
        if env_name == 'MultiKeyNav':
            self.fc1 = nn.Linear(7, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_mu = nn.Linear(128, embedding_dim)       
            self.fc3_log_sigma = nn.Linear(128, embedding_dim)    
        elif env_name == 'CartPoleVar':
            self.fc1 = nn.Linear(7, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_mu = nn.Linear(128, embedding_dim)       
            self.fc3_log_sigma = nn.Linear(128, embedding_dim)       
        elif env_name == 'PointMass':
            self.fc1 = nn.Linear(7, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_mu = nn.Linear(128, embedding_dim)       
            self.fc3_log_sigma = nn.Linear(128, embedding_dim)    
        elif env_name == 'BasicKarel':
            self.fc1 = nn.Linear(88, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_mu = nn.Linear(128, embedding_dim)       
            self.fc3_log_sigma = nn.Linear(128, embedding_dim)            

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.fc3_mu(x)
        log_sigma = self.fc3_log_sigma(x)
        sigma = torch.exp(log_sigma)
        
        q_z = torch.distributions.Normal(loc=mu, scale=sigma)
        z = q_z.rsample()

        return z, q_z

class DynamicsNetwork(nn.Module):
    def __init__(self, env_name, embedding_dim):
        super(DynamicsNetwork, self).__init__()
        if env_name == 'MultiKeyNav':
            self.fc1 = nn.Linear(embedding_dim + 5 + 7, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_s_n = nn.Linear(128, 5)
            self.fc3_r = nn.Linear(128, 1)
        elif env_name == 'CartPoleVar':
            self.fc1 = nn.Linear(embedding_dim + 5 + 1, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_s_n = nn.Linear(128, 5)
            self.fc3_r = nn.Linear(128, 1)    
        elif env_name == 'PointMass':
            self.fc1 = nn.Linear(embedding_dim + 4 + 2, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_s_n = nn.Linear(128, 4)
            self.fc3_r = nn.Linear(128, 1)  
        elif env_name == 'BasicKarel':
            self.fc1 = nn.Linear(embedding_dim + 52 + 6, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3_s_n = nn.Linear(128, 52)
            self.fc3_r = nn.Linear(128, 1)        

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        s_n = self.fc3_s_n(x)
        r = self.fc3_r(x)

        return s_n, r    
