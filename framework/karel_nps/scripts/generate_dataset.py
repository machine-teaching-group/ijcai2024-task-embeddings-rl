# External imports
import json
import torch
import os
from tqdm import tqdm
import argparse
import sys
import random
from itertools import chain, zip_longest

from tqdm import tqdm

if __package__ is None:
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from karel.consistency import Simulator

def translate(seq,
              vocab):
    return [vocab[str(elt)] for elt in seq]


def load_input_file(path_to_dataset, path_to_vocab):
    '''
    path_to_dataset: File containing the data
    path_to_vocab: File containing the vocabulary
    '''
    tgt_tkn2idx = {
        '<pad>': 0,
    }
    next_id = 1
    with open(path_to_vocab, 'r') as vocab_file:
        for line in vocab_file.readlines():
            tgt_tkn2idx[line.strip()] = next_id
            next_id += 1
    tgt_idx2tkn = {}
    for tkn, idx in tgt_tkn2idx.items():
        tgt_idx2tkn[idx] = tkn

    vocab = {"idx2tkn": tgt_idx2tkn,
             "tkn2idx": tgt_tkn2idx}
    
    print("idx2tkn:", tgt_idx2tkn)

    path_to_ds_cache = path_to_dataset.replace('.json', '.thdump')
    
    with open(path_to_dataset, 'r') as dataset_file:
        tgts = []
        data = []
        for sample_str in tqdm(dataset_file.readlines()):
            sample_data = json.loads(sample_str)

            # Get the target program
            tgt_program_tkn = sample_data['program_tokens']
            tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
            
            data.append(sample_data)
            tgts.append(tgt_program_idces)

    dataset = {"complete_samples": data,
                "targets": tgts}

    return dataset, vocab

def get_sample_depth(tokens):
    depth = 0
    max_depth = 0
    for tkn in tokens:
        if tkn in ["m(", "w(", "i(", "e(", "r("]:
            depth += 1
        elif tkn in ["m)", "w)", "i)", "e)", "r)"]:
            if depth > max_depth:
                max_depth = depth
            depth -= 1
            if depth < 0:
                depth = 0
    return max_depth

def get_sample_num_blocks(tokens):
    return len(tokens) 

def get_type(tokens):
    # A B: Loop_Present Conditionals_Present

    A = 0
    B = 0
    for tkn in tokens:
        if tkn in ['REPEAT', 'WHILE']:
            A = 1
        if tkn in ['IF', 'IFELSE', 'ELSE']:
            B = 1       
    return A, B      

def filter_tasks(dataset, vocab):
    # 1: (NA + C + L) no masking (Normal + Loops + Conditionals)
    # 2: (NA + C) mask loops (Normal + Conditionals)
    # 3: (NA + L) mask conditionals (Normal + Loops)
    # 4: (NA) mask loops and conditionals (Normal)
    samples_1 = []
    samples_2 = []
    samples_3 = []
    samples_4 = []
    val_samples_1 = []
    val_samples_2 = []
    val_samples_3 = []
    val_samples_4 = []
    simulator = Simulator(vocab["idx2tkn"])
    i = 0

    NA_train = {10: 12692, 11: 2940, 12: 1492, 13: 781, 14: 460}
    L_train = {10: 4000, 11: 4000, 12: 4000, 13: 4000, 14: 4000}
    C_train = {10: 0, 11: 0, 12: 0, 13: 7736, 14: 12636}

    NA_val = {10: 3172, 11: 735, 12: 373, 13: 195, 14: 115}
    L_val = {10: 1000, 11: 1000, 12: 1000, 13: 1000, 14: 1000}
    C_val = {10: 0, 11: 0, 12: 0, 13: 1933, 14: 3158}

    for sample, target in tqdm(zip(dataset["complete_samples"], dataset["targets"]), total=len(dataset["complete_samples"])):
        tokens = simulator.tkn_prog_from_idx(target)
        d = get_sample_depth(tokens)
        n = get_sample_num_blocks(tokens)
        if d <= 2 and  10 <= n <= 14:
            A, B = get_type(tokens)
            if A == 0 and B == 0:
                if NA_train[n] > 0:
                    NA_train[n] -= 1
                    samples_1.append(sample)
                    samples_2.append(sample)
                    samples_3.append(sample)
                    samples_4.append(sample)
                elif NA_val[n] > 0:
                    NA_val[n] -= 1
                    val_samples_1.append(sample)
                    val_samples_2.append(sample)
                    val_samples_3.append(sample)
                    val_samples_4.append(sample)
            elif A == 0 and B == 1:
                if C_train[n] > 0:
                    C_train[n] -= 1
                    samples_1.append(sample)
                    samples_2.append(sample)
                elif C_val[n] > 0:
                    C_val[n] -= 1
                    val_samples_1.append(sample)
                    val_samples_2.append(sample)
            elif A == 1 and B == 0:
                if L_train[n] > 0:
                    L_train[n] -= 1
                    samples_1.append(sample)
                    samples_3.append(sample)
                elif L_val[n] > 0:
                    L_val[n] -= 1
                    val_samples_1.append(sample)
                    val_samples_3.append(sample)

    train_samples_1 = samples_1
    train_samples_2 = samples_2
    train_samples_3 = samples_3
    train_samples_4 = samples_4

    print('NA_train:', NA_train)
    print('L_train:', L_train)
    print('C_train:', C_train)

    print('NA_val:', NA_val)
    print('L_val:', L_val)
    print('C_val:', C_val)    

    return train_samples_1, val_samples_1, train_samples_2, val_samples_2, train_samples_3, val_samples_3, train_samples_4, val_samples_4

def shuffle_dataset(dataset, batch_size, randomize=True):
    '''
    We are going to group together samples that have a similar length, to speed up training
    batch_size is passed so that we can align the groups
    '''
    pairs = list(zip(dataset["complete_samples"], dataset["targets"]))
    bucket_fun = lambda x: len(x[1]) // 5
    pairs.sort(key=bucket_fun, reverse=True)
    grouped_pairs = [pairs[pos: pos + batch_size]
                     for pos in range(0,len(pairs), batch_size)]
    if randomize:
        to_shuffle = grouped_pairs[:-1]
        random.shuffle(to_shuffle)
        grouped_pairs[:-1] = to_shuffle
    pairs = chain.from_iterable(grouped_pairs)
    in_seqs, out_seqs = zip(*pairs)
    return {
        "complete_samples": in_seqs,
        "targets": out_seqs
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate a smaller dataset.')

    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--vocabulary_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--validation_output_path', type=str)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args() 

    dataset, vocab = load_input_file(args.dataset_path, args.vocabulary_path)
    dataset = shuffle_dataset(dataset, args.batch_size)
    samples_list_1, val_samples_list_1, samples_list_2, val_samples_list_2, samples_list_3, val_samples_list_3, samples_list_4, val_samples_list_4 = \
        filter_tasks(dataset, vocab)

    with open(f'{args.output_path}_1.json', 'w') as fp:
        for sample in samples_list_1:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.validation_output_path}_1.json', 'w') as fp:
        for sample in val_samples_list_1:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.output_path}_2.json', 'w') as fp:
        for sample in samples_list_2:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.validation_output_path}_2.json', 'w') as fp:
        for sample in val_samples_list_2:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.output_path}_3.json', 'w') as fp:
        for sample in samples_list_3:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.validation_output_path}_3.json', 'w') as fp:
        for sample in val_samples_list_3:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.output_path}_4.json', 'w') as fp:
        for sample in samples_list_4:
            fp.write(json.dumps(sample) + '\n')

    with open(f'{args.validation_output_path}_4.json', 'w') as fp:
        for sample in val_samples_list_4:
            fp.write(json.dumps(sample) + '\n')
