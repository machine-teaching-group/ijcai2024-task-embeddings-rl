import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Generate a dataset of trajectories.")
    
    parser.add_argument('--device', type=str, default='cpu')
    
    parser.add_argument('--prefix', type=str, default="Default")
    parser.add_argument('--expert_policy_path', type=str, default="runs_train_expert")
    
    parser.add_argument('--env_spec_path', type=str, default="specs/MultiKeyNav_spec.json")
    
    parser.add_argument('--num_trajectories', type=int, default=10000)

    config = parser.parse_args()
    
    config.device = torch.device(config.device)

    return config
