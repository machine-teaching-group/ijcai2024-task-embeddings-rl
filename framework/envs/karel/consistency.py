import torch

from karel.ast import Ast
from karel.ast_converter import AstParser, AstParseException, AstConverter
from karel.fast_emulator import FastEmulator

class Simulator(object):

    def __init__(self, idx_to_token_vocab):
        super(Simulator, self).__init__()
        self.idx_to_token = idx_to_token_vocab
        self.ast_parser = AstParser()
        self.ast_converter = AstConverter()
        self.emulator = FastEmulator(max_ticks=200)

        self.prog_start = ['DEF', 'run', 'm(']
        self.prog_end = ['m)']

    def tkn_prog_from_idx(self, prg_idxs):
        return [self.idx_to_token[idx] for idx in prg_idxs]
    
    def tkn_from_idx(self, idx):
        return self.idx_to_token[idx]

    def get_prog_ast(self, prg_idxs):
        prg_tkns = self.tkn_prog_from_idx(prg_idxs)
        try:
            prg_ast_json = self.ast_parser.parse(prg_tkns)
        except AstParseException as e:
            return False, None
        prog_ast = Ast(prg_ast_json)
        return True, prog_ast

    def get_prog_tkns(self, ast_json):
        ast = Ast(ast_json)
        prg_tkns = self.ast_converter.to_tokens(ast)
        return prg_tkns

    def simple_prog_from_target(self, target, input_world):
        _, prog_ast = self.get_prog_ast(target)
        res_emu = self.run_prog(prog_ast, input_world)
        actions_list = res_emu.actions

        prog = self.prog_start + actions_list + self.prog_end

        return prog

    def run_cf_once(self, cf, inp_grid):
        emu_result = self.emulator.emulate_cf_once(cf, inp_grid)
        return emu_result

    def run_prog(self, prog_ast, inp_grid):
        emu_result = self.emulator.emulate(prog_ast, inp_grid)
        return emu_result
