import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Generate data to train the embedding network.")
    
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--prefix', type=str, default="Default")
    parser.add_argument('--exp', type=str, default="1")
    parser.add_argument('--policies_path', type=str, default="runs_policy")
    parser.add_argument('--expert_policy_path', type=str, default="runs_train_expert")
    
    parser.add_argument('--env_spec_path', type=str, default="specs/SimpleKarel_spec.json")
    
    parser.add_argument('--max_episode_len', type=int, default=20)
    
    # MI
    parser.add_argument('--num_samples_1', type=int, default=100)
    # Norm
    parser.add_argument('--num_samples_2', type=int, default=10)
    
    parser.add_argument('--train_size', type=int, default=6000)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=2000)

    config = parser.parse_args()
    
    config.device = torch.device(config.device)

    return config
