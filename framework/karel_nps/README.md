# Karel NPS

This codebase builds upon the repository available at [repo](https://github.com/bunelr/GandRL_for_NPS), which has been ported to Python3 and adapted for our framework. Refer to the original repository for more details and setup instructions.

## Contents

1. [Dataset Generation](#Dataset-Generation)
2. [Population Generation](#Population-Generation)

## Dataset Generation

Generate a smaller dataset of tasks from the Karel dataset:

  ```bash
  python3 scripts/generate_dataset.py --dataset_path /path/to/dataset_dir/train.json --vocabulary_path /path/to/dataset_dir/vocab.vocab --output_path /path/to/output_dir/train --validation_output_path /path/to/output_dir/val --batch_size 128
  ```

The data generation script also produces datasets for the subpopulations.

## Population Generation

Generate a population of agents:

  ```bash
  python cmds/train_cmd.py --kernel_size 3 --conv_stack "64,64,64" --fc_stack "512" --tgt_embedding_size 256 --lstm_hidden_size 256 --nb_lstm_layers 2 --signal supervised --nb_ios 5 --nb_epochs 100 --optim_alg Adam --batch_size 128 --learning_rate 1e-4 --val_frequency 1 --snapshot_delta 0.1 --train_file /path/to/generated_dataset/train_1.json --val_file /path/to/generated_dataset/val_1.json --vocab /path/to/generated_dataset/vocab.vocab --result_folder ICLR18Exps/pop --use_grammar --use_cuda
  python cmds/train_cmd.py --kernel_size 3 --conv_stack "64,64,64" --fc_stack "512" --tgt_embedding_size 256 --lstm_hidden_size 256 --nb_lstm_layers 2 --signal supervised --nb_ios 5 --nb_epochs 100 --optim_alg Adam --batch_size 128 --learning_rate 1e-4 --val_frequency 1 --snapshot_delta 0.1 --train_file /path/to/generated_dataset/train_2.json --val_file /path/to/generated_dataset/val_2.json --vocab /path/to/generated_dataset/vocab.vocab --result_folder ICLR18Exps/pop_m_loops --use_grammar --use_cuda
  python cmds/train_cmd.py --kernel_size 3 --conv_stack "64,64,64" --fc_stack "512" --tgt_embedding_size 256 --lstm_hidden_size 256 --nb_lstm_layers 2 --signal supervised --nb_ios 5 --nb_epochs 100 --optim_alg Adam --batch_size 128 --learning_rate 1e-4 --val_frequency 1 --snapshot_delta 0.1 --train_file /path/to/generated_dataset/train_3.json --val_file /path/to/generated_dataset/val_3.json --vocab /path/to/generated_dataset/vocab.vocab --result_folder ICLR18Exps/pop_m_conditionals --use_grammar --use_cuda
  python cmds/train_cmd.py --kernel_size 3 --conv_stack "64,64,64" --fc_stack "512" --tgt_embedding_size 256 --lstm_hidden_size 256 --nb_lstm_layers 2 --signal supervised --nb_ios 5 --nb_epochs 100 --optim_alg Adam --batch_size 128 --learning_rate 1e-4 --val_frequency 1 --snapshot_delta 0.1 --train_file /path/to/generated_dataset/train_4.json --val_file /path/to/generated_dataset/val_4.json --vocab /path/to/generated_dataset/vocab.vocab --result_folder ICLR18Exps/pop_m_loops_conditionals --use_grammar --use_cuda
  ```


