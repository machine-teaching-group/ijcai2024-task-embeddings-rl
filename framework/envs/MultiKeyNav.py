from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random
from copy import deepcopy

class MultiKeyNavEnv(Env):
    def __init__(self, p_rand=0, delta_a_1=0.075, delta_a_2=0.01, failure_gamma=0.999):
        self.p_rand = p_rand
        self.delta_a_1 = delta_a_1
        self.delta_a_2 = delta_a_2
        self.failure_gamma = failure_gamma
        self.action_space = Discrete(5)
        # 0: Left, 1: Right, 2: Pick Key A, 3: Pick Key B, 4: Pick Key C, 5: Pick Key D, 6: Finish 
        self.observation_space = Dict({"location": Box(
            low=0, high=1, shape=(1,), dtype=np.float32), 
                                  "key_A_status": Discrete(2),
                                  "key_B_status": Discrete(2),
                                  "key_C_status": Discrete(2),
                                  "key_D_status": Discrete(2),
                                  "door_type_bit_1": Discrete(2),
                                  "door_type_bit_2": Discrete(2)})
        # door types: 00: A B, 01: A C, 10: B D, 11: C D 
        self.state = self.observation_space.sample()

    def step(self, action):
        if random.random() > self.failure_gamma:
            done = True
            reward = 0
            info = {}

            return self.state, reward, done, info
        
        if random.random() < self.p_rand:
            action = self.action_space.sample()

        done = False
        reward = 0
        
        # Indices: A:1, B:2, C:3, D:4
        s_idx_A = 1
        s_idx_B = 2
        s_idx_C = 3
        s_idx_D = 4
        
        # Segments: A: 0-0.1, B: 0.2-0.3, C: 0.4-0.5, D: 0.6-0.7
        seg_A = [0, 0.1]
        seg_B = [0.2, 0.3]
        seg_C = [0.4, 0.5]
        seg_D = [0.6, 0.7]
        
        # Pick Action Indices: A:2, B:3, C:4, D:5
        idx_A = 2
        idx_B = 3
        idx_C = 4
        idx_D = 5
        
        key_1_index = None
        key_2_index = None
        req_keys = None
        
        if self.state[5] == 0 and self.state[6] == 0:
            key_1_index, key_2_index = s_idx_A, s_idx_B
            req_keys = ['A', 'B']
        elif self.state[5] == 0 and self.state[6] == 1:
            key_1_index, key_2_index = s_idx_A, s_idx_C
            req_keys = ['A', 'C']
        elif self.state[5] == 1 and self.state[6] == 0:
            key_1_index, key_2_index = s_idx_B, s_idx_D
            req_keys = ['B', 'D']
        elif self.state[5] == 1 and self.state[6] == 1:
            key_1_index, key_2_index = s_idx_C, s_idx_D
            req_keys = ['C', 'D']
            
        if action == idx_A:
            # pick key_A
            if self.state[0] <= seg_A[1] and self.state[0] >= seg_A[0] and self.state[s_idx_A] == 0:
                self.state[s_idx_A] = 1
            else:
                done = True
        elif action == idx_B:
            # pick key_B
            if self.state[0] <= seg_B[1] and self.state[0] >= seg_B[0] and self.state[s_idx_B] == 0:
                self.state[s_idx_B] = 1
            else:
                done = True
        elif action == idx_C:
            # pick key_C
            if self.state[0] <= seg_C[1] and self.state[0] >= seg_C[0] and self.state[s_idx_C] == 0:
                self.state[s_idx_C] = 1
            else:
                done = True
        elif action == idx_D:
            # pick key_D
            if self.state[0] <= seg_D[1] and self.state[0] >= seg_D[0] and self.state[s_idx_D] == 0:
                self.state[s_idx_D] = 1
            else:
                done = True                    
        elif action == 0:
            # move left
            new_state = self.state[0] - self.delta_a_1 + random.uniform(-self.delta_a_2, self.delta_a_2)
            if new_state < 0:
                pass
            else:
                self.state[0] = new_state
        elif action == 1:
            # move right
            new_state = self.state[0] + self.delta_a_1 + random.uniform(-self.delta_a_2, self.delta_a_2)
            if new_state > 1:
                pass
            else:
                self.state[0] = new_state    
        else:
            # finish
            if self.state[0] >= 0.9 and self.state[key_1_index] == 1 and self.state[key_2_index] == 1:
                done = True
                reward = 1
            else:
                done = True
                pass       

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass
    
    def _state_dict_to_list(self, state):
        return np.asarray([state["location"][0], state["key_A_status"], state["key_B_status"], state["key_C_status"], state["key_D_status"], state["door_type_bit_1"], state["door_type_bit_2"]])

    def reset(self, state=None):
        if state is not None:
            self.state = deepcopy(state)
            return state

        self.state = self._state_dict_to_list(self.observation_space.sample())

        return self.state