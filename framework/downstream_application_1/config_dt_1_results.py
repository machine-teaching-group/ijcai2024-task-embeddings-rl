import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Results for downstream task 1 (Agent's Performance Prediction).")
    
    parser.add_argument('--device', type=str, default='cpu')
    
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--prefix', type=str, default="Default")
    parser.add_argument('--data_path', type=str, default="runs_generate_dt_1_data")
    parser.add_argument('--env_spec_path', type=str, default="specs/SimpleKarel_spec.json")
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--quiz_size', type=int, default=50)
    parser.add_argument('--sub_size', type=int, default=1)
    
    # Ours
    parser.add_argument('--embedding_dim', type=int, default=1)
    parser.add_argument('--embedding_net_path', type=str, default="runs_train_embedding")
    
    # PredModel
    parser.add_argument('--embedding_dim_pred', type=int, default=1)
    
    config = parser.parse_args()
    
    config.device = torch.device(config.device)

    return config
