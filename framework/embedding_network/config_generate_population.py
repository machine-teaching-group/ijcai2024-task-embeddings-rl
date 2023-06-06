import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Generate agent population.")
    
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--prefix', type=str, default="Default")
    
    parser.add_argument('--env_spec_path', type=str, default="specs/SimpleKarel_spec.json")
    
    parser.add_argument('--technique', type=str, default="il")
    
    parser.add_argument('--action_mask', type=str, default=None)
    
    # rl
    #-------
    parser.add_argument('--use_SAC', action='store_true')
    # REINFORCE
    parser.add_argument('--num_episodes', type=int, default=20000)
    # SAC
    parser.add_argument('--total_timesteps', type=int, default=5000000)
    #-------
    
    # il
    #-------
    parser.add_argument('--input_path', type=str, default="bc_data/SimpleKarel_data.json")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=30000)
    #-------
    
    parser.add_argument('--pretrained_path', type=str, default=None)
    
    parser.add_argument('--max_episode_len', type=int, default=20)
    
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--snapshot_delta', type=float, default=0.001)
    parser.add_argument('--num_rollouts_per_task', type=int, default=1)
    
    parser.add_argument('--random_seed', type=int, default=0)

    config = parser.parse_args()
    
    config.device = torch.device(config.device)

    return config
