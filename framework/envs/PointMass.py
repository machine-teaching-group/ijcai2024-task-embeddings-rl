import numpy as np
import random
from gym import Env, spaces
from copy import deepcopy
import torch

class PointMassEnv(Env):

    def __init__(self, left=None, failure_gamma=1):
        self.action_space = spaces.Box(np.array([-10., -10.]), np.array([10., 10.]))
        self.observation_space = spaces.Box(np.array([-4., -np.inf, -4., -np.inf, -4, 0.5, 0]),
                                            np.array([4., np.inf, 4., np.inf, 4, 8, 4]))
        self.failure_gamma = failure_gamma
        self._state = None
        self._goal_state = np.array([0., 0., -3., 0.])
        self._dt = 0.01
        self.H = 100
        self.left = left

    def reset(self, state=None):
        self.h = 0
        if state is not None:
            self._state = deepcopy(state)
            self._gate = np.asarray([state[-3], 0])
            return np.copy(self._state)
        
        self._state = np.array([0., 0., 3., 0.,\
            np.random.uniform(-4, 4), np.random.uniform(0.5, 8), np.random.uniform(0, 4)])
        
        if self.left is not None:
            if self.left:
                while True:
                    if self._state[-3] + 0.5 * self._state[-2] < 0:
                        break
                    self._state = np.array([0., 0., 3., 0.,\
                        np.random.uniform(-4, 4), np.random.uniform(0.5, 8), np.random.uniform(0, 4)])
            else:
                while True:
                    if self._state[-3] - 0.5 * self._state[-2] > 0:
                        break
                    self._state = np.array([0., 0., 3., 0.,\
                        np.random.uniform(-4, 4), np.random.uniform(0.5, 8), np.random.uniform(0, 4)])    
            
        self._gate = np.asarray([self._state[-3], 0])
        
        return np.copy(self._state)

    def _step_internal(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_der = np.zeros(4 + 3)
        state_der[:4][0::2] = state[:4][1::2]
        friction_param = state[-1]
        state_der[:4][1::2] = 1.5 * action - friction_param * state[:4][1::2] + np.random.normal(0, 0.05, (2,))
        new_state = np.clip(state + self._dt * state_der, self.observation_space.low,
                            self.observation_space.high)

        crash = False
        if state[2] >= 0 > new_state[2] or state[2] <= 0 < new_state[2]:
            alpha = (0. - state[2]) / (new_state[2] - state[2])
            x_crit = alpha * new_state[0] + (1 - alpha) * state[0]

            if np.abs(x_crit - state[-3]) > 0.5 * state[-2]:
                new_state = np.array([x_crit, 0., 0., 0., state[-3], state[-2], state[-1]])
                crash = True

        return new_state, crash

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        self.h += 1
        
        if random.random() > self.failure_gamma:
            done = True
            reward = 0
            info = {"success": False}

            return self._state, reward, done, info

        action = np.clip(action, self.action_space.low, self.action_space.high)

        new_state = self._state
        crash = False
        for i in range(0, 10):
            new_state, crash = self._step_internal(new_state, action)
            if crash:
                break

        self._state = np.copy(new_state)

        info = {"success": np.linalg.norm(self._goal_state[0::2] - new_state[:4][0::2]) < 0.25}
        
        reward = 0
        if info["success"]:
            reward += 1e4
        
        if new_state[2] < 0:
            reward += 0.001 * np.exp(-0.6 * np.linalg.norm(self._goal_state[0::2] - new_state[:4][0::2]))
        else:
            reward += 0.0001 * np.exp(-0.6 * np.linalg.norm(self._gate - new_state[:4][0::2])) 

        return new_state, reward, self.h > self.H or crash or info["success"], info
    
    def rollout(self, model, state):
        with torch.no_grad():
            state = self.reset(state=state)
            while True:
                action, _ = model.predict(state)
                state, _, done, info = self.step(action)
                if done or info["success"]:
                    return info["success"]