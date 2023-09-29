## Contents

0. [Pre-requisites](#Pre-requisites)
1. [Visualizations](#Visualizations)
2. [Formats](#Formats)
3. [General Instructions](#General-Instructions)
4. [Instructions: MultiKeyNav Environment](#MultiKeyNav-Environment)
5. [Instructions: CartPoleVar Environment](#CartPoleVar-Environment)
5. [Instructions: PointMass Environment](#PointMass-Environment)
5. [Instructions: Karel Environment](#Karel-Environment)
6. [Instructions: BasicKarel Environment](#BasicKarel-Environment)

## Note

Our code for the *Karel* environment builds upon the repository available at [repo](https://github.com/bunelr/GandRL_for_NPS).

## Pre-requisites

This codebase requires Python 3, [gym](https://github.com/openai/gym#installation), [NumPy](https://numpy.org/install/), [PyTorch](https://pytorch.org/get-started/locally/), [scikit-learn](https://scikit-learn.org/stable/install.html), [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html), [Matplotlib](https://matplotlib.org/stable/users/getting_started/), [SciencePlots](https://github.com/garrettj403/SciencePlots#getting-started), and [tqdm](https://github.com/tqdm/tqdm#installation).

## Visualizations

The visualization plots in the paper (Fig. 3 and 4) could be reproduced as follows:

1. (Fig. 3a) *MultiKeyNav*

   ```bash
   python3 vis_embedding_scripts/fig_3a.py
   ```

2. (Fig. 3b) *CartPoleVar*

   ```bash
   python3 vis_embedding_scripts/fig_3b.py
   ```

3. (Fig. 3c) *PointMass*

   ```bash
   python3 vis_embedding_scripts/fig_3c.py
   ```
   
3. (Fig. 3e) *BasicKarel*

   ```bash
   python3 vis_embedding_scripts/fig_3e.py
   ```
   
4. (Fig. 4a) *Without norm constraints*

   ```bash
   python3 vis_embedding_scripts/fig_4a.py
   ```
   
5. (Fig. 4b) *pickKey actions are masked*

   ```bash
   python3 vis_embedding_scripts/fig_4b.py
   ```
   
6. (Fig. 4c) *All doors: KeyA, KeyB*

   ```bash
   python3 vis_embedding_scripts/fig_4c.py
   ```
   
7. (Fig. 4d) *All doors: KeyA*

   ```bash
   python3 vis_embedding_scripts/fig_4d.py
   ```   

## Formats

### Environment

The environment class definition file should be structured similar to the following example: `MultiKeyNav.py` contains the definition for the class `MultiKeyNavEnv`, and `MultiKeyNavEnv` extends `gym.Env`. Alternatively, you could introduce an if-clause inside the `parse_env_spec` function in `utils.py`.

### Specification

The specification file is a JSON file with the following format:
  
   ```
   {
   "env_name": str,
   "performance_eval_tasks": list[{'state': list[float]}]
   }
   ```
   
### Behavioral Cloning Data

The data for behavioral cloning should be prepared as a JSON file with the following format:

   ```
   list[{'s': list[float], 'a': int}]
   ```

## General Instructions

### Preparation

1. Create a class definition file for the environment in the `envs` directory.

2. Create a specification file for the environment in the `specs` directory.

3. (**OPTIONAL**) Prepare data for behavioral cloning.

### Execute the End-to-End Pipeline

1. Train an expert policy:

   ```bash
   python3 embedding_network/train_expert.py
   ```

2. Generate a population of agents:

   ```bash
   python3 embedding_network/generate_population.py
   ```
   
3. Generate a set of ordinal constraints:

   ```bash
   python3 embedding_network/generate_ordinal_constraints.py
   ```

4. Train the embedding network:

   ```bash
   python3 embedding_network/train_embedding_network.py
   ```

5. Collect transitions from the environment:

   ```bash
   python3 pred_model_baseline/gen_data.py
   ```   

6. Preprocess the collected transitions:

   ```bash
   python3 pred_model_baseline/build_dataset.py
   ```  

7. Train the predictive model baseline:

   ```bash
   python3 pred_model_baseline/train.py
   ```  
   
8. Generate the benchmarks for the downstream applications:

   ```bash
   python3 downstream_application_1/generate_dt_1_data.py
   python3 downstream_application_2/generate_dt_2_data.py
   ```
   
6. Evaluate the techniques on the benchmarks:

   ```bash
   python3 downstream_application_1/dt_1_results_kfold.py
   python3 downstream_application_2/dt_2_results.py
   python3 downstream_application_2/dt_2_results_pred_model.py
   python3 downstream_application_2/dt_2_results_vis_sim.py
   python3 downstream_application_2/dt_2_results_edit_distance.py
   python3 downstream_application_2/dt_2_results_opt.py
   ```
   
## MultiKeyNav Environment

### Preparation

1. Create a specification file for the environment in the `specs` directory:

   ```bash
   python3 prep_scripts/MultiKeyNav/generate_specs.py
   ```

2. Prepare data for pretraining the policy using behavioral cloning:

   ```bash
   python3 prep_scripts/MultiKeyNav/generate_bc_data.py
   ```
   
### Execute the End-to-End Pipeline

1. Train an expert policy:

   ```bash
   python3 embedding_network/train_expert.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_default_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --num_rollouts_per_task 10
   ```

2. Generate a population of agents:

   ```bash
   python3 embedding_network/generate_population.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_default_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_maskA_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_maskB_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_maskC_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_maskD_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --technique "il" --input_path "bc_data/MultiKeyNav_maskAllKeys_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 40 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   ```
   
3. Generate a set of ordinal constraints:

   ```bash
   python3 embedding_network/generate_ordinal_constraints.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --max_episode_len 40 --num_samples_1 100 --num_samples_2 10 --train_size 5000 --val_size 1000 --test_size 1000
   ```

4. Train the embedding network:

   ```bash
   python3 embedding_network/train_embedding_network.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --embedding_dim 6 --batch_size 128 --num_epochs 300 --log_interval 1 --val_interval 1 --lr 0.001 --alpha 0.4 --device "cuda"
   ```

5. Collect transitions from the environment:

   ```bash
   python3 pred_model_baseline/gen_data.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --num_trajectories 10000
   ```   

6. Preprocess the collected transitions:

   ```bash
   python3 pred_model_baseline/build_dataset.py --env_spec_path "specs/MultiKeyNav_spec.json"
   ```  

7. Train the predictive model baseline:

   ```bash
   python3 pred_model_baseline/train.py --env_spec_path "specs/MultiKeyNav_spec.json" --embedding_dim 6 --batch_size 512 --num_epochs 500 --log_interval 1 --val_interval 1 --lr 0.001 --alpha_1 1 --alpha_2 1 --alpha_3 0.01 --device "cuda"
   ```
   
8. Generate the benchmarks for the downstream applications:

   ```bash
   python3 downstream_application_1/generate_dt_1_data.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --max_episode_len 40 --quiz_size 20 --train_size 5000 --test_size 5000 --num_samples_1 500 --num_samples_2 10 --num_samples_3 10 --num_samples_4 10000
   python3 downstream_application_2/generate_dt_2_data.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --max_episode_len 40 --num_options 10 --train_size 200 --test_size 200 --num_samples_1 100 --num_samples_2 10
   python3 downstream_application_2/generate_reference_tasks.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --max_episode_len 40 --num_reference_tasks 5 --pool_size 500 --num_samples_2 10
   ```
   
9. Evaluate the techniques on the benchmarks:

   ```bash
   python3 downstream_application_1/dt_1_results_kfold.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --num_splits 10 --quiz_size 20 --sub_size 10 --embedding_dim 6 --embedding_dim_pred 6
   python3 downstream_application_2/dt_2_results.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --num_options 10 --start_idx 0 --num_examples 50 --embedding_dim 6 --k 3
   python3 downstream_application_2/dt_2_results_pred_model.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --embedding_dim_pred 6 --num_options 10 --start_idx 0 --num_examples 50 --k 3
   python3 downstream_application_2/dt_2_results_vis_sim.py --prefix "MultiKeyNavExps" --num_options 10 --start_idx 0 --num_examples 50 --k 3
   python3 downstream_application_2/dt_2_results_edit_distance.py --prefix "MultiKeyNavExps" --num_options 10 --start_idx 0 --num_examples 50 --k 3
   python3 downstream_application_2/dt_2_results_opt.py --prefix "MultiKeyNavExps" --env_spec_path "specs/MultiKeyNav_spec.json" --num_options 10 --start_idx 0 --num_examples 50 --num_samples_1 100 --num_samples_2 10 --max_episode_len 40 --pop_frac 1 --k 3
   ```  
   
## CartPoleVar Environment

### Preparation

1. Create a specification file for the environment in the `specs` directory:

   ```bash
   python3 prep_scripts/CartPoleVar/generate_spec.py
   ```

2. Prepare data for pretraining the policy using behavioral cloning:

   ```bash
   python3 prep_scripts/CartPoleVar/generate_bc_data.py
   ```
   
### Execute the End-to-End Pipeline

1. Train an expert policy:

   ```bash
   python3 embedding_network/train_expert.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --technique "il" --input_path "bc_data/CartPoleVar_default_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 200 --log_interval 500 --lr 0.001 --num_rollouts_per_task 10
   ```

2. Generate a population of agents:

   ```bash
   python3 embedding_network/generate_population.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --technique "il" --input_path "bc_data/CartPoleVar_default_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 200 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 1
   python3 embedding_network/generate_population.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --technique "il" --input_path "bc_data/CartPoleVar_bias_1.json" --batch_size 512 --num_epochs 20000 --max_episode_len 200 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 1
   python3 embedding_network/generate_population.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --technique "il" --input_path "bc_data/CartPoleVar_bias_2.json" --batch_size 512 --num_epochs 20000 --max_episode_len 200 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 1
   python3 embedding_network/generate_population.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --technique "il" --input_path "bc_data/CartPoleVar_bias_3.json" --batch_size 512 --num_epochs 20000 --max_episode_len 200 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 1
   python3 embedding_network/generate_population.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --technique "il" --input_path "bc_data/CartPoleVar_bias_4.json" --batch_size 512 --num_epochs 20000 --max_episode_len 200 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 1
   ```
   
3. Generate a set of ordinal constraints:

   ```bash
   python3 embedding_network/generate_ordinal_constraints.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --max_episode_len 200 --num_samples_1 1 --num_samples_2 1 --train_size 5000 --val_size 1000 --test_size 1000
   ```

4. Train the embedding networks:

   ```bash
   python3 embedding_network/train_embedding_network.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --embedding_dim 3 --batch_size 128 --num_epochs 500 --log_interval 1 --val_interval 1 --lr 0.001 --alpha 0.4 --device "cuda"
   ```

5. Collect transitions from the environment:

   ```bash
   python3 pred_model_baseline/gen_data.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --num_trajectories 10000
   ```   

6. Preprocess the collected transitions:

   ```bash
   python3 pred_model_baseline/build_dataset.py --env_spec_path "specs/CartPoleVar_spec.json"
   ```  

7. Train the predictive model baseline:

   ```bash
   python3 pred_model_baseline/train.py --env_spec_path "specs/CartPoleVar_spec.json" --embedding_dim 3 --batch_size 512 --num_epochs 500 --log_interval 1 --val_interval 1 --lr 0.001 --alpha_1 1 --alpha_2 1 --alpha_3 0.01 --device "cuda"
   ```     
   
8. Generate the benchmarks for the downstream applications:

   ```bash
   python3 downstream_application_1/generate_dt_1_data.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --max_episode_len 200 --quiz_size 20 --train_size 5000 --test_size 5000 --num_samples_1 500 --num_samples_2 1 --num_samples_3 1 --num_samples_4 10000
   python3 downstream_application_2/generate_dt_2_data.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --max_episode_len 200 --num_options 10 --train_size 200 --test_size 200 --num_samples_1 1 --num_samples_2 1
   python3 downstream_application_2/generate_reference_tasks.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --max_episode_len 200 --num_reference_tasks 5 --pool_size 500 --num_samples_2 1
   ```
   
9. Evaluate the techniques on the benchmarks:

   ```bash
   python3 downstream_application_1/dt_1_results_kfold.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --num_splits 10 --quiz_size 20 --sub_size 10 --embedding_dim 3 --embedding_dim_pred 3
   python3 downstream_application_2/dt_2_results.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --num_options 10 --embedding_dim 3 --k 3
   python3 downstream_application_2/dt_2_results_pred_model.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --embedding_dim_pred 3 --num_options 10 --start_idx 0 --num_examples 50 --k 3
   python3 downstream_application_2/dt_2_results_vis_sim.py --prefix "CartPoleVarExps" --num_options 10 --start_idx 0 --num_examples 50 --k 3 --num_examples 50
   python3 downstream_application_2/dt_2_results_edit_distance.py --prefix "CartPoleVarExps" --num_options 10 --start_idx 0 --num_examples 50 --k 3
   python3 downstream_application_2/dt_2_results_opt.py --prefix "CartPoleVarExps" --env_spec_path "specs/CartPoleVar_spec.json" --num_options 10 --start_idx 0 --num_examples 50 --num_samples_1 1 --num_samples_2 1 --max_episode_len 200 --pop_frac 1 --k 3
   ```    

## PointMass Environment

### Preparation

1. Create a specification file for the environment in the `specs` directory:

   ```bash
   python3 prep_scripts/PointMass/generate_spec.py
   ```
   
### Execute the End-to-End Pipeline

1. Generate a population of agents:

   ```bash
   python3 embedding_network/generate_population.py --prefix "PointMassExps" --env_spec_path "specs/PointMass_spec.json" --technique "rl" --batch_size 32 --total_timesteps 5000000 --log_interval 50000 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10 --random_seed 0 --use_SAC
   python3 embedding_network/generate_population.py --prefix "PointMassExps" --env_spec_path "specs/PointMass_Left_spec.json" --technique "rl" --batch_size 32 --total_timesteps 5000000 --log_interval 50000 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10 --random_seed 100 --use_SAC
   python3 embedding_network/generate_population.py --prefix "PointMassExps" --env_spec_path "specs/PointMass_Right_spec.json" --technique "rl" --batch_size 32 --total_timesteps 5000000 --log_interval 50000 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10 --random_seed 1000 --use_SAC
   ```
   
2. Generate a set of ordinal constraints:

   ```bash
   python3 embedding_network/generate_ordinal_constraints.py --prefix "PointMassExps" --env_spec_path "specs/PointMass_spec.json" --num_samples_1 100 --num_samples_2 10 --train_size 5000 --val_size 1000 --test_size 1000
   ```

3. Train the embedding network:

   ```bash
   python3 embedding_network/train_embedding_network.py --prefix "PointMassExps" --env_spec_path "specs/PointMass_spec.json" --embedding_dim 3 --batch_size 128 --num_epochs 300 --log_interval 1 --val_interval 1 --lr 0.001 --alpha 0.4 --device "cuda" 
   ``` 

4. Collect transitions from the environment:

   ```bash
   python3 pred_model_baseline/gen_data.py --prefix "PointMassExps" --env_spec_path "specs/PointMass_spec.json" --num_trajectories 10000
   ```   

5. Preprocess the collected transitions:

   ```bash
   python3 pred_model_baseline/build_dataset.py --env_spec_path "specs/PointMass_spec.json"
   ```  

6. Train the predictive model baseline:

   ```bash
   python3 pred_model_baseline/train.py --env_spec_path "specs/PointMass_spec.json" --embedding_dim 3 --batch_size 512 --num_epochs 500 --log_interval 1 --val_interval 1 --lr 0.001 --alpha_1 1 --alpha_2 1 --alpha_3 0.01 --device "cuda"
   ```  

## Karel Environment

### Preparation

1. Create a specification file for the environment in the `specs` directory:

   ```bash
   python3 prep_scripts/ICLR18/generate_spec.py 
   ```
   
### Execute the End-to-End Pipeline

1. Refer to the README in the `karel_nps` directory for instructions on generating a population of agents.

2. Update the path to the dataset and the vocabulary in `envs/ICLR18.py`.
   
3. Generate a set of ordinal constraints:

   ```bash
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "1"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "2"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "3"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "4"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "5"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "6"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "7"
   python3 embedding_network/generate_ordinal_constraints.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --num_samples_1 1 --num_samples_2 1 --train_size 10000 --val_size 2000 --test_size 2000 --exp "8"
   python3 embedding_network/postprocess_ordinal_constraints.py --prefix "ICLR18Exps" --train_size 10000 --val_size 2000 --test_size 2000
   ```

4. Train the embedding network:

   ```bash
   python3 embedding_network/train_embedding_network.py --prefix "ICLR18Exps" --env_spec_path "specs/ICLR18_spec.json" --embedding_dim 2 --batch_size 512 --num_epochs 10 --log_interval 1 --val_interval 1 --lr 0.0001 --alpha 0.4 --device "cuda" 
   ```    
   
## BasicKarel Environment

### Preparation

1. Create a specification file for the environment in the `specs` directory:

   ```bash
   python3 prep_scripts/BasicKarel/generate_spec.py
   ```

2. Prepare data for pretraining the policy using behavioral cloning:

   ```bash
   python3 prep_scripts/BasicKarel/generate_bc_data.py
   ```
   
### Execute the End-to-End Pipeline

1. Train an expert policy:

   ```bash
   python3 embedding_network/train_expert.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --technique "il" --input_path "bc_data/BasicKarel_default_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 20 --log_interval 500 --lr 0.001 --num_rollouts_per_task 10
   ```

2. Generate a population of agents:

   ```bash
   python3 embedding_network/generate_population.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --technique "il" --input_path "bc_data/BasicKarel_default_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 20 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --technique "il" --input_path "bc_data/BasicKarel_mask_pick_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 20 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --technique "il" --input_path "bc_data/BasicKarel_mask_put_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 20 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   python3 embedding_network/generate_population.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --technique "il" --input_path "bc_data/BasicKarel_mask_pick_put_data.json" --batch_size 512 --num_epochs 20000 --max_episode_len 20 --log_interval 500 --lr 0.001 --snapshot_delta 0.01 --num_rollouts_per_task 10
   ```
   
3. Generate a set of ordinal constraints:

   ```bash
   python3 embedding_network/generate_ordinal_constraints.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --max_episode_len 20 --num_samples_1 100 --num_samples_2 10 --train_size 10000 --val_size 2000 --test_size 2000
   ```

4. Train the embedding network:

   ```bash
   python3 embedding_network/train_embedding_network.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --embedding_dim 1 --batch_size 128 --num_epochs 300 --log_interval 1 --val_interval 1 --lr 0.001 --alpha 0 --device "cuda"
   ``` 

5. Collect transitions from the environment:

   ```bash
   python3 pred_model_baseline/gen_data.py --prefix "BasicKarelExps" --env_spec_path "specs/BasicKarel_spec.json" --num_trajectories 10000
   ```   

6. Preprocess the collected transitions:

   ```bash
   python3 pred_model_baseline/build_dataset.py --env_spec_path "specs/BasicKarel_spec.json"
   ```  

7. Train the predictive model baseline:

   ```bash
   python3 pred_model_baseline/train.py --env_spec_path "specs/BasicKarel_spec.json" --embedding_dim 8 --batch_size 512 --num_epochs 500 --log_interval 1 --val_interval 1 --lr 0.001 --alpha_1 1 --alpha_2 1 --alpha_3 0.01 --device "cuda"
   ```  
