import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Train expert policy.")
    
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--prefix', type=str, default="Default")
    
    parser.add_argument('--env_spec_path', type=str, default="specs/LineKeyNav_spec.json")
    
    parser.add_argument('--technique', type=str, default="il")
    
    # rl
    #-------
    parser.add_argument('--num_episodes', type=int, default=20000)
    #-------
    
    # il
    #-------
    parser.add_argument('--input_path', type=str, default="bc_data/LineKeyNav_data.json")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=50000)
    #-------
    
    parser.add_argument('--pretrained_path', type=str, default=None)
    
    parser.add_argument('--max_episode_len', type=int, default=50)
    
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--num_rollouts_per_task', type=int, default=1)
    
    parser.add_argument('--random_seed', type=int, default=0)

    config = parser.parse_args()
    
    config.device = torch.device(config.device)

    return config
