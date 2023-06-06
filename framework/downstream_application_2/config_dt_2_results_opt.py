import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Results for downstream task 2 (Task Selection).")
    
    parser.add_argument('--device', type=str, default='cpu')
    
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--prefix', type=str, default="Default")
    
    parser.add_argument('--env_spec_path', type=str, default="specs/SimpleKarel_spec.json")

    parser.add_argument('--data_path', type=str, default="runs_generate_dt_2_data")
    parser.add_argument('--num_options', type=int, default=10)
    
    # MI estimation
    parser.add_argument('--num_samples_1', type=int, default=100)
    # performance estimation
    parser.add_argument('--num_samples_2', type=int, default=10)
    
    parser.add_argument('--max_episode_len', type=int, default=20)
    
    parser.add_argument('--policies_path', type=str, default="runs_policy")
    
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_examples', type=int, default=200)
    parser.add_argument('--pop_frac', type=float, default=1)
    parser.add_argument('--k', type=int, default=1)
    
    config = parser.parse_args()
    
    config.device = torch.device(config.device)


    return config
