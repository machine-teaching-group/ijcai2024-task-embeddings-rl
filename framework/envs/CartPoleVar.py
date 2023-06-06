"""
Adapted from: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

from gym import Env
from gym.spaces import Discrete, Box, Dict
import math
import numpy as np
import random
from copy import deepcopy

class CartPoleVarEnv(Env):
    def __init__(self, failure_gamma=1):
        self.failure_gamma = failure_gamma
        self.action_space = Discrete(2)
        # 0: move cart to the left, 1: move cart to the right
         
        # dummy space 
        self.observation_space = Box(
            low=-1, high=1, shape=(7,), dtype=np.float32)
        
        self.observation_space_sampling = Dict({"cart_position": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "cart_velocity": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "pole_angle": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "pole_angular_velocity": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "force_magnitude": Box(
            low=5, high=15, shape=(1,), dtype=np.float32),
                                       "task_type": Box(
            low=0, high=1, shape=(1,), dtype=np.int),
                                       "num_steps": Discrete(200)})                                      
        
        self.key2idx = {"cart_position": 0, "cart_velocity": 1, "pole_angle": 2, "pole_angular_velocity": 3, "force_magnitude": 4, "task_type": 5, "num_steps": 6}
        
        self.tau = 0.02  # seconds between state updates
        self.x_threshold = 2.4

        self.state = self.reset()

        self.required_steps = 200

    def step(self, action):
        if random.random() > self.failure_gamma:
            done = True
            reward = 0
            info = {}

            return self.state, reward, done, info
        
        self.state[self.key2idx["num_steps"]] += 1
        
        x = self.state[self.key2idx["cart_position"]]
        x_dot = self.state[self.key2idx["cart_velocity"]]
        theta = self.state[self.key2idx["pole_angle"]]
        theta_dot = self.state[self.key2idx["pole_angular_velocity"]]
        gravity = 9.8
        masscart = 1
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5
        polemass_length = masspole * length
        force_mag = self.state[self.key2idx["force_magnitude"]]
        task_type = self.state[self.key2idx["task_type"]]
        theta_threshold_radians = 12 * 2 * math.pi / 360
        
        if force_mag == 0:
            f = 10
        else:
            f = force_mag
        force = f * (1 - 2 * task_type) if action == 1 else - f * (1 - 2 * task_type)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + polemass_length * theta_dot**2 * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state[self.key2idx["cart_position"]] = x
        self.state[self.key2idx["cart_velocity"]] = x_dot
        self.state[self.key2idx["pole_angle"]] = theta
        self.state[self.key2idx["pole_angular_velocity"]] = theta_dot

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians
        )
    
        reward = 0

        if self.state[self.key2idx["num_steps"]] >= self.required_steps:
            done = True
            reward = 1
            
        info = {}

        return self.state, reward, done, info
    
    def _state_dict_to_list(self, state):
        return np.asarray([state["cart_position"][0], state["cart_velocity"][0], state["pole_angle"][0], state["pole_angular_velocity"][0], state["force_magnitude"][0], state["task_type"][0], int(state["num_steps"])])

    def reset(self, state=None):
        if state is not None:
            self.state = deepcopy(state)
            return state
        
        state = self.observation_space_sampling.sample()
        if np.random.random() >= 0.5:
            state["force_magnitude"] *= -1
        state["num_steps"] = 0
        state["task_type"][0] = int(np.random.random() >= 0.5)
        self.state = self._state_dict_to_list(state)
        
        return self.state

class CartPoleOriginalEnv(Env):
    def __init__(self, failure_gamma=1):
        self.failure_gamma = failure_gamma
        self.action_space = Discrete(2)
        # 0: move cart to the left, 1: move cart to the right
         
        # dummy space 
        self.observation_space = Box(
            low=-1, high=1, shape=(7,), dtype=np.float32)
        
        self.observation_space_sampling = Dict({"cart_position": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "cart_velocity": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "pole_angle": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "pole_angular_velocity": Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32),
                                       "force_magnitude": Box(
            low=5, high=15, shape=(1,), dtype=np.float32),
                                       "task_type": Box(
            low=0, high=0, shape=(1,), dtype=np.int),
                                       "num_steps": Discrete(200)})                                      
        
        self.key2idx = {"cart_position": 0, "cart_velocity": 1, "pole_angle": 2, "pole_angular_velocity": 3, "force_magnitude": 4, "task_type": 5, "num_steps": 6}
        
        self.tau = 0.02  # seconds between state updates
        self.x_threshold = 2.4

        self.state = self.reset()

        self.required_steps = 200

    def step(self, action):
        if random.random() > self.failure_gamma:
            done = True
            reward = 0
            info = {}

            return self.state, reward, done, info
        
        self.state[self.key2idx["num_steps"]] += 1
        
        x = self.state[self.key2idx["cart_position"]]
        x_dot = self.state[self.key2idx["cart_velocity"]]
        theta = self.state[self.key2idx["pole_angle"]]
        theta_dot = self.state[self.key2idx["pole_angular_velocity"]]
        gravity = 9.8
        masscart = 1
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5
        polemass_length = masspole * length
        force_mag = self.state[self.key2idx["force_magnitude"]]
        task_type = self.state[self.key2idx["task_type"]]
        theta_threshold_radians = 12 * 2 * math.pi / 360
        
        if force_mag == 0:
            f = 10
        else:
            f = force_mag
        force = f * (1 - 2 * task_type) if action == 1 else - f * (1 - 2 * task_type)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + polemass_length * theta_dot**2 * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state[self.key2idx["cart_position"]] = x
        self.state[self.key2idx["cart_velocity"]] = x_dot
        self.state[self.key2idx["pole_angle"]] = theta
        self.state[self.key2idx["pole_angular_velocity"]] = theta_dot

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians
        )
    
        reward = 1 if not done else 0

        if self.state[self.key2idx["num_steps"]] >= self.required_steps:
            done = True
            reward = 1
            
        info = {}

        return self.state, reward, done, info
    
    def _state_dict_to_list(self, state):
        return np.asarray([state["cart_position"][0], state["cart_velocity"][0], state["pole_angle"][0], state["pole_angular_velocity"][0], state["force_magnitude"][0], state["task_type"][0], int(state["num_steps"])])

    def reset(self, state=None):
        if state is not None:
            self.state = deepcopy(state)
            return state
        
        state = self.observation_space_sampling.sample()
        state["num_steps"] = 0
        self.state = self._state_dict_to_list(state)
        
        return self.state        