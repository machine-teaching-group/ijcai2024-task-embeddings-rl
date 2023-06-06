from gym import Env
from gym.spaces import Discrete
import numpy as np
import random
from copy import deepcopy
import json
from glob import glob

class BasicKarelEnv(Env):
    def __init__(self, data_path="envs/BasicKarelData/data/train", failure_gamma=0.999):
        self.data_path = data_path
        self._load_data()
        
        self.dim = 4
        self.failure_gamma = failure_gamma
        self.action_space = Discrete(6)
        # action: 0: move, 1: turnLeft, 2: turnRight, 3: pickMarker, 4: putMarker, 5: finish

        # Dummy space
        self.observation_space = Discrete(1)

        self.dir_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        self.rev_dir_map = {0: 'north', 1: 'south', 2: 'east', 3: 'west'}
        self.directions = {"north": np.asarray([-1, 0]), "south": np.asarray([1, 0]), "east": np.asarray([0, 1]), "west": np.asarray([0, -1])}
        self.left_turns = {"north": "west", "south": "east", "east": "north", "west": "south"}
        self.right_turns = {"north": "east", "south": "west", "east": "south", "west": "north"}
        
        self.state = {"pregrid_agent_row": None,
                      "pregrid_agent_col": None,
                      "pregrid_agent_dir": None,
                      "postgrid_agent_row": None,
                      "postgrid_agent_col": None,
                      "postgrid_agent_dir": None,
                      "walls": None,
                      "pregrid_markers": None,
                      "postgrid_markers": None}            
        
    def _load_json(self, file_path):
        fd = open(file_path)
        example_data = json.load(fd)
        fd.close()
    
        return example_data    
    
    def _load_data(self):
        self.data_size = len(glob(f'{self.data_path}/task/*.json'))
        self.tasks = []
        self.seqs = []
        
        for i in range(self.data_size):
            self.tasks.append(self._load_json(f'{self.data_path}/task/{i}_task.json'))
            self.seqs.append(self._load_json(f'{self.data_path}/seq/{i}_seq.json')["sequence"])
        
    def _get_bitmaps(self, walls, pregrid_agent_row, pregrid_agent_column, pregrid_agent_dir, pregrid_markers, postgrid_agent_row, postgrid_agent_column, postgrid_agent_dir, postgrid_markers):
        bitmaps = np.zeros((88,))

        wall_map = np.zeros((4, 4))
        if len(walls):
            wall_map[tuple(np.asarray(walls).T)] = 1
        bitmaps[0:16] = wall_map.reshape(-1,)

        pregrid_marker_map = np.zeros((4, 4))
        if len(pregrid_markers):
            pregrid_marker_map[tuple(np.asarray(pregrid_markers).T)] = 1
        bitmaps[16:32] = pregrid_marker_map.reshape(-1,)

        postgrid_marker_map = np.zeros((4, 4))
        if len(postgrid_markers):
            postgrid_marker_map[tuple(np.asarray(postgrid_markers).T)] = 1
        bitmaps[32:48] = postgrid_marker_map.reshape(-1,)

        pregrid_location_map = np.zeros((4, 4))
        pregrid_location_map[pregrid_agent_row, pregrid_agent_column] = 1
        bitmaps[48:64] = pregrid_location_map.reshape(-1,)

        postgrid_location_map = np.zeros((4, 4))
        postgrid_location_map[postgrid_agent_row, postgrid_agent_column] = 1
        bitmaps[64:80] = postgrid_location_map.reshape(-1,)

        bitmaps[80 + self.dir_map[pregrid_agent_dir]] = 1
        bitmaps[84 + self.dir_map[postgrid_agent_dir]] = 1
    
        return bitmaps
                    
    def step(self, action):
        if random.random() > self.failure_gamma:
            done = True
            reward = 0
            info = {}

            return self.state_to_bitmaps(self.state), reward, done, info

        done = False
        reward = 0

        if action == 0:
            new_position = np.asarray([self.state["pregrid_agent_row"], self.state["pregrid_agent_col"]]) \
                + self.directions[self.state["pregrid_agent_dir"]]
            if (new_position[0] < 0 or new_position[0] >= self.dim) or (new_position[1] < 0 or new_position[1] >= self.dim):
                done = True
            elif new_position.tolist() in self.state["walls"]:
                done = True    
            else:
                self.state["pregrid_agent_row"], self.state["pregrid_agent_col"] = int(new_position[0]), int(new_position[1])
        elif action == 1:
            self.state["pregrid_agent_dir"] = self.left_turns[self.state["pregrid_agent_dir"]]
        elif action == 2:
            self.state["pregrid_agent_dir"] = self.right_turns[self.state["pregrid_agent_dir"]]
        elif action == 3:
            try:
                self.state["pregrid_markers"].remove([self.state["pregrid_agent_row"], self.state["pregrid_agent_col"]])
            except ValueError:
                done = True    
        elif action == 4:
            if [self.state["pregrid_agent_row"], self.state["pregrid_agent_col"]] in self.state["pregrid_markers"]:
                done = True
            else:
                self.state["pregrid_markers"].append([self.state["pregrid_agent_row"], self.state["pregrid_agent_col"]]) 
        elif action == 5:
            done = True
            self.state["pregrid_markers"].sort()
            self.state["postgrid_markers"].sort()
            
            if self.state["pregrid_agent_row"] == self.state["postgrid_agent_row"] \
                and self.state["pregrid_agent_col"] == self.state["postgrid_agent_col"] \
                    and self.state["pregrid_agent_dir"] == self.state["postgrid_agent_dir"] \
                        and self.state["pregrid_markers"] == self.state["postgrid_markers"]:
                            reward = 1

        info = {}

        return self.state_to_bitmaps(self.state), reward, done, info
    
    def state_to_bitmaps(self, state):
        bitmaps = self._get_bitmaps(state["walls"], state["pregrid_agent_row"], state["pregrid_agent_col"],\
                state["pregrid_agent_dir"], state["pregrid_markers"], state["postgrid_agent_row"], state["postgrid_agent_col"],\
                state["postgrid_agent_dir"], state["postgrid_markers"])
        
        return bitmaps
    
    def bitmaps_to_state(self, bitmaps):
        state = {"pregrid_agent_row": None,
                 "pregrid_agent_col": None,
                 "pregrid_agent_dir": None,
                 "postgrid_agent_row": None,
                 "postgrid_agent_col": None,
                 "postgrid_agent_dir": None,
                 "walls": [],
                 "pregrid_markers": [],
                 "postgrid_markers": []}
        
        wall_map = bitmaps[0:16].reshape(4, 4)
        idx_i, idx_j = np.where(wall_map)
        for i, j in zip(idx_i, idx_j):
            state["walls"].append([i, j])
        
        pregrid_marker_map = bitmaps[16:32].reshape(4, 4)
        idx_i, idx_j = np.where(pregrid_marker_map)
        for i, j in zip(idx_i, idx_j):
            state["pregrid_markers"].append([i, j])
        
        postgrid_marker_map = bitmaps[32:48].reshape(4, 4)
        idx_i, idx_j = np.where(postgrid_marker_map)
        for i, j in zip(idx_i, idx_j):
            state["postgrid_markers"].append([i, j])
        
        pregrid_location_map = bitmaps[48:64].reshape(4, 4)
        idx_i, idx_j = np.where(pregrid_location_map)
        state["pregrid_agent_row"], state["pregrid_agent_col"] = idx_i[0], idx_j[0]
        
        postgrid_location_map = bitmaps[64:80].reshape(4, 4)
        idx_i, idx_j = np.where(postgrid_location_map)
        state["postgrid_agent_row"], state["postgrid_agent_col"] = idx_i[0], idx_j[0]

        state["pregrid_agent_dir"] = self.rev_dir_map[np.where(bitmaps[80:84])[0][0]]
        state["postgrid_agent_dir"] = self.rev_dir_map[np.where(bitmaps[84:])[0][0]]
        
        assert self.state_to_bitmaps(state).tolist() == bitmaps.tolist() 
        
        return state

    def render(self):
        pass

    def reset(self, state=None):
        if state is None:
            idx = np.random.randint(0, self.data_size)
            self.state = deepcopy(self.tasks[idx])
            
            return self.state_to_bitmaps(self.state)
        
        if isinstance(state, dict):
            self.state = deepcopy(state)

            return self.state_to_bitmaps(self.state)
        else:
            self.state = self.bitmaps_to_state(state)
        
        return deepcopy(state)