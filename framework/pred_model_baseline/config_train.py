import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Train the embedding network.")
    
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--env_spec_path', type=str, default="specs/MultiKeyNav_spec.json")
    parser.add_argument('--embedding_dim', type=int, default=2)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--alpha_1', type=float, default=1)
    parser.add_argument('--alpha_2', type=float, default=1)
    parser.add_argument('--alpha_3', type=float, default=1)

    parser.add_argument('--random_seed', type=int, default=0)

    config = parser.parse_args()
    
    config.device = torch.device(config.device)

    return config
