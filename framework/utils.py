import json
import importlib
import numpy as np
import torch
import gym

def load_json(file_path):
    """
    Reads data from file_path and returns a JSON object.
    :param file_path: str
    :returns: JSON
    """   
    fd = open(file_path)
    data = json.load(fd)
    fd.close()
    
    return data

def dump_json(data, file_path):
    """
    Dumps the the JSON object on file_path.
    :param data: JSON
    :param file_path: str
    """
    fd = open(file_path, 'w')
    json.dump(data, fd, indent=1)
    fd.close()

def parse_env_spec(env_spec_path, return_env_class=False):
    """
    Reads and parses the environment specification file.
    :param env_spec_path: str
    :returns: 
        - gym.Env
        - list[numpy.ndarray(float)]
    """   
    env_spec = load_json(env_spec_path)
    
    env_name = env_spec['env_name']

    env_module = importlib.import_module(f'envs.{env_name}')
    env_class = getattr(env_module, f'{env_name}Env')
    if 'env_params' in env_spec:
        env = env_class(*env_spec['env_params'])
    else:
        env = env_class()   
    
    performance_eval_tasks = []
    for task_spec in env_spec['performance_eval_tasks']:
        performance_eval_tasks.append(np.asarray(task_spec['state']))
        
    if return_env_class:
        return env_name, env_class, env, performance_eval_tasks    
    else:
        return env_name, env, performance_eval_tasks  

def rollout(policy, s, env, max_episode_len, use_env_fn=False):
    """
    Performs a rollout of the policy from state s on env and returns the success flag.
    :param policy: torch.nn.Module
    :param s: numpy.ndarray(float)
    :param env: gym.Env
    :param max_episode_len: int
    :param use_env_fn: bool
    :returns: bool
    """
    if use_env_fn:
        return env.rollout(policy, state=s)

    with torch.no_grad():
        state = env.reset(state=s)
        for _ in range(max_episode_len):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            if done:
                if reward == 1:
                    return True
                else:
                    return False

    return False

def get_trajectory(policy, s, env, max_episode_len, use_env_fn=False):
    """
    Performs a rollout of the policy from state s on env and returns a list of ('s', 'a') pairs.
    :param policy: torch.nn.Module
    :param s: numpy.ndarray(float)
    :param env: gym.Env
    :param max_episode_len: int
    :param use_env_fn: bool
    :returns: list[dict]
    """
    if use_env_fn:
        return None

    trajectory = []
    with torch.no_grad():
        state = env.reset(state=s)
        for _ in range(max_episode_len):
            action = policy.select_action(state)
            trajectory.append({'s': state.tolist(), 'a': action})
            state, _, done, _ = env.step(action)
            if done:
                break

    return trajectory

def get_trajectory_trans(policy, env_name, env, max_episode_len, one_hot_action=None, n_context_idx=None):
    """
    Performs a rollout of the policy from a random task on env and returns a list of transitions ('s', 'a', 'r', 's_n').
    :param policy: torch.nn.Module
    :param env_name: str
    :param env: gym.Env
    :param max_episode_len: int
    :param one_hot_action: bool
    :param n_context_idx: list[int]
    :returns: list[dict]
    """
    solved = False
    trajectory = []
    with torch.no_grad():
        state = env.reset()
        s_0 = state.tolist()
        for _ in range(max_episode_len):
            if env_name == 'PointMass':
                action, _ = policy.predict(state)
            else:
                action = policy.select_action(state)
            if one_hot_action is not None:
                action_rep = [0 for _ in range(one_hot_action)]
                action_rep[action] = 1
            else:
                action_rep = action
                if env_name == 'PointMass':
                    action_rep = action_rep.tolist()  
            transition = {'s': state[n_context_idx].tolist() if n_context_idx is not None else state.tolist(), 'a': action_rep}
            state, reward, done, info = env.step(action)
            transition['s_n'] = state[n_context_idx].tolist() if n_context_idx is not None else state.tolist()
            if done:
                if env_name == 'PointMass':
                    solved = info["success"]
                elif reward == 1:
                    solved = True
                transition['r'] = 1 if solved else 0    
                trajectory.append(transition)    
                break  
            transition['r'] = 0
            trajectory.append(transition)  

    return s_0, trajectory, solved

