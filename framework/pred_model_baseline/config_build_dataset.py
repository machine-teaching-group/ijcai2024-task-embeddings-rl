import torch
import argparse

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="Build the dataset to train the baseline.")

    parser.add_argument('--env_spec_path', type=str, default="specs/MultiKeyNav_spec.json")

    config = parser.parse_args()

    return config
