import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from config_generate_ordinal_constraints import get_config
import numpy as np
from glob import glob
from tqdm import tqdm
from utils import load_json

def postprocess(config):
    """
    Combines ordinal constraints and stores them as .npy files.
    """ 
    net_dir = os.path.split(os.path.realpath(__file__))[0]
    
    num_parts_MI = len(glob(f'{net_dir}/runs_generate_MI_data/{config.prefix}/MI_OrdinalConstraints/data_train_*.json'))
    num_parts_Norm = len(glob(f'{net_dir}/runs_generate_MI_data/{config.prefix}/Norm_OrdinalConstraints/data_train_*.json'))
    
    #for constraint_type in [['MI', num_parts_MI], ['Norm', num_parts_Norm]]:
    for constraint_type in [['Norm', num_parts_Norm]]:
        for set_type in [['train', config.train_size], ['val', config.val_size], ['test', config.test_size]]:
            first_itr = True
            count = 0
            for file in tqdm(glob(f'{net_dir}/runs_generate_MI_data/{config.prefix}/{constraint_type[0]}_OrdinalConstraints/data_{set_type[0]}_*.json')):
                data_part = load_json(file)
                if first_itr:
                    first_itr = False
                    state_shape = np.asarray(data_part[0]['s_1']['s']).shape
                    data_np = np.zeros((constraint_type[1] * set_type[1], 3 if constraint_type[0] == 'MI' else 2, *(state_shape)))
                for example in tqdm(data_part):
                    data_np[count][0] = np.asarray(example['s_1']['s'])
                    data_np[count][1] = np.asarray(example['s_2']['s'])
                    if constraint_type == 'MI':
                        data_np[count][2] = np.asarray(example['s_3']['s'])
                    count += 1  
            np.save(f'{net_dir}/runs_generate_MI_data/{config.prefix}/{constraint_type[0]}_OrdinalConstraints/data_{set_type[0]}_np.npy', data_np)

if __name__ == "__main__":
    config = get_config()

    postprocess(config)
