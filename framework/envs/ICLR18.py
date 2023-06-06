from gym import Env
from gym.spaces import Discrete
import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
import json
import torch
from torch.autograd import Variable
import sys
import os

parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent_dir)

from nps.data import load_input_file, get_minibatch, shuffle_dataset
from karel.consistency import Simulator
from syntax.checker import PySyntaxChecker

class ICLR18Env(Env):
    def __init__(self, dataset_path="envs/data/karel/train_1.json", \
        vocabulary_path="envs/data/karel/vocab.vocab"):
        self.dataset_path = dataset_path
        self.vocabulary_path = vocabulary_path
        self._setup()
        
        self.nb_ios = 5
        self.beam_size = 10
        
        # Dummy Action Space
        self.action_space = Discrete(1)
        # Dummy Observation Space
        self.observation_space = Discrete(1)
        
        self.state = self.observation_space.sample()  
        
    def _setup(self):
        self.dataset, self.vocab = load_input_file(self.dataset_path, self.vocabulary_path)
        self.tgt_start = self.vocab["tkn2idx"]["<s>"]
        self.tgt_end = self.vocab["tkn2idx"]["m)"]
        self.tgt_pad = self.vocab["tkn2idx"]["<pad>"]
        
        self.simulator = Simulator(self.vocab["idx2tkn"])
        
        self.syntax_checker = PySyntaxChecker(self.vocab["tkn2idx"], False)

    def get_depth(self, tokens):
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
        
    def get_type(self, state):
        sp_idx = int(state[0])
        target = self.dataset["targets"][sp_idx]
        tokens = self.simulator.tkn_prog_from_idx(target)
        
        # A B: Loop_Present Conditionals_Present

        A = 0
        B = 0
        for tkn in tokens:
            if tkn in ['REPEAT', 'WHILE']:
                A = 1
            if tkn in ['IF', 'IFELSE', 'ELSE']:
                B = 1       
        return (A, B, self.get_depth(tokens))    
        
    def step(self, action):
        pass

    def render(self):
        pass

    def reset(self, state=None):
        if state is not None:
            pass
        
        sp_idx = random.randint(0, len(self.dataset["sources"]))
        
        if not (sp_idx < len(self.dataset["sources"])):
            print('Avoided error!')
            sp_idx = len(self.dataset["sources"]) - 1

        inp_grids, out_grids, \
        in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
        inp_worlds, out_worlds, \
        targets, \
        inp_test_worlds, out_test_worlds = get_minibatch(self.dataset, sp_idx, 1,
                                                         self.tgt_start, self.tgt_end, self.tgt_pad,
                                                         self.nb_ios, shuffle=False, volatile_vars=True)
        
        max_len = out_tgt_seq.size(1) + 10
        
        # (1, 5, 16, 18, 18) * 2 -> (51840) + 1 {sp_idx, inp_grids, out_grids}-> (51841)
        
        return np.concatenate([np.asarray(sp_idx).reshape(1,), inp_grids.flatten().numpy(), out_grids.flatten().numpy()])
    
    def rollout(self, model, state):
        sp_idx = int(state[0])
        model.set_syntax_checker(self.syntax_checker)
        
        inp_grids, out_grids, \
        in_tgt_seq, in_tgt_seq_list, out_tgt_seq, \
        inp_worlds, out_worlds, \
        _, \
        inp_test_worlds, out_test_worlds = get_minibatch(self.dataset, sp_idx, 1,
                                                         self.tgt_start, self.tgt_end, self.tgt_pad,
                                                         self.nb_ios, shuffle=False, volatile_vars=True)
        
        max_len = out_tgt_seq.size(1) + 10
        
        decoded = model.beam_sample(inp_grids, out_grids,
                                    self.tgt_start, self.tgt_end, max_len,
                                    self.beam_size, 1)
        
        for batch_idx, (target, sp_decoded,
                        sp_input_worlds, sp_output_worlds,
                        sp_test_input_worlds, sp_test_output_worlds) in \
            enumerate(zip(out_tgt_seq.chunk(out_tgt_seq.size(0)), decoded,
                          inp_worlds, out_worlds,
                          inp_test_worlds, out_test_worlds)):
            
            target = target.cpu().data.squeeze().numpy().tolist()
            target = [tkn_idx for tkn_idx in target if tkn_idx != self.tgt_pad]
            
            for rank, dec in enumerate(sp_decoded):
                pred = dec[-1]
                parse_success, cand_prog = self.simulator.get_prog_ast(pred)
                if (not parse_success):
                    continue
                generalizes = True
                for (input_world, output_world) in zip(sp_input_worlds, sp_output_worlds):
                    res_emu = self.simulator.run_prog(cand_prog, input_world)
                    if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                        # This prediction is semantically incorrect.
                        generalizes = False
                        break
                for (input_world, output_world) in zip(sp_test_input_worlds, sp_test_output_worlds):
                    res_emu = self.simulator.run_prog(cand_prog, input_world)
                    if (res_emu.status != 'OK') or res_emu.crashed or (res_emu.outgrid != output_world):
                        # This prediction is semantically incorrect.
                        generalizes = False
                        break
                if generalizes:
                    # Score for all the following ranks
                    for top_idx in range(rank, 1):
                        return True
                    break  
                
            return False