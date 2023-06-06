import numpy as np
import random
from utils import rollout
from copy import deepcopy

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True)

def compute_binary_entropy(p):
    """
    Computes the entropy of a bernoulli random variable with parameter p
    :param p: float
    :returns: float
    """ 
    ent = (- p * np.log2(p) - (1 - p) * np.log2(1 - p)) if p not in [0, 1] else 0 
    assert not np.isnan(ent), f'{ent}, {p}'
    return ent

def estimate_MI(s_i, s_j, env, policies, max_episode_len, num_samples, use_env_fn=False):
    """
    Estimates the mutual information between O_{s_{i}} and O_{s_{j}}
    :param s_i: numpy.ndarray(float)
    :param s_j: numpy.ndarray(float)
    :param env: gym.Env
    :param policies: list[torch.nn.Module]
    :param max_episode_len: int
    :param num_samples: int
    :returns: float
    """ 
    n_i = 0
    n_j = 0
    n_i_j_1 = 0
    n_i_j_0 = 0
    
    for policy_org in policies:
        policy = deepcopy(policy_org)
        for _ in range(num_samples):
            success_i = rollout(policy, s_i, env, max_episode_len, use_env_fn)
            success_j = rollout(policy, s_j, env, max_episode_len, use_env_fn)

            if success_i:
                n_i += 1
                if success_j:
                    n_i_j_1 += 1
                else:
                    n_i_j_0 += 1
            if success_j:
                n_j += 1
                
    total_samples = len(policies) * num_samples            
    p_Oi_1 = np.divide(n_i, total_samples)
    p_Oi_1 = p_Oi_1 if not np.isnan(p_Oi_1) else 0
    H_Oi = compute_binary_entropy(p_Oi_1)
    term_1 = H_Oi

    p_Oi_Oj_1_1 = np.divide(n_i_j_1, n_j)
    p_Oi_Oj_1_1 =  p_Oi_Oj_1_1 if not np.isnan(p_Oi_Oj_1_1) else 0
    H_Oi_Oj_1 = compute_binary_entropy(p_Oi_Oj_1_1)
    p_Oi_Oj_1_0 = np.divide(n_i_j_0, total_samples - n_j)
    p_Oi_Oj_1_0 =  p_Oi_Oj_1_0 if not np.isnan(p_Oi_Oj_1_0) else 0
    H_Oi_Oj_0 = compute_binary_entropy(p_Oi_Oj_1_0)
    p_Oj_1 = np.divide(n_j, total_samples)
    p_Oj_1 =  p_Oj_1 if not np.isnan(p_Oj_1) else 0
    term_2 = p_Oj_1 * H_Oi_Oj_1 + (1 - p_Oj_1) * H_Oi_Oj_0
    assert not np.isnan(term_1), f"{n_i}, {n_j}, {n_i_j_1}, {n_i_j_0}"
    assert not np.isnan(term_2)
    return term_1 - term_2
