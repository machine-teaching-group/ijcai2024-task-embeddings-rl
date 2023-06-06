import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
from envs.BasicKarel import BasicKarelEnv
from models import EmbeddingNetwork
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

plt.style.use('science')

EMBEDDING_NET_PATH = 'vis_embedding_scripts/embedding_networks/embedding_model_withoutNormConstraints_BasicKarel_dim1.pt'

embedding_dim = 1

def get_color(bitmaps, env):
    state = env.bitmaps_to_state(np.asarray(bitmaps))
    l_1 = len(state["pregrid_markers"])
    l_2 = len(state["postgrid_markers"])
    
    if l_1 == l_2:
        # Nav
        return 'red'
    elif l_1 > l_2:
        # Pick
        return 'blue'
    else:
        # Put
        return 'green'  

def main():
    env = BasicKarelEnv()
    
    model_data = torch.load(EMBEDDING_NET_PATH, map_location=torch.device('cpu'))
    embedding_model = EmbeddingNetwork('BasicKarel', embedding_dim)
    embedding_model.load_state_dict(model_data['parameters'])
    embedding_model.eval()  
    
    random.seed(0)
    np.random.seed(0)
        
    tasks = np.load('vis_embedding_scripts/BasicKarel_tasks.npy')    
    tasks = [(t, get_color(t, env), 'o') for t in tasks]

    for x, color, marker in tasks:
        x_b = torch.FloatTensor(x).view(1, -1)
        with torch.no_grad():
            embedding = embedding_model(x_b)
            norm = np.linalg.norm(embedding)  
        task_types = {'red': 'Navigation', 'blue': 'Pick Marker(s)', 'green': 'Pick Marker(s)'}
        plt.scatter(embedding[0][0], 0, marker=marker, s=norm*50, c=(color), label=task_types[color], alpha=0.3, edgecolors='black')
    
    plt.xticks([])
    plt.yticks([])
    
    f = lambda c: plt.plot([],[], marker='o', color=c, ls="none", alpha=0.6)[0]
    
    colors = ['blue', 'red', 'green']
    
    handles = [f(colors[i]) for i in range(3)]

    labels = ['$\mathrm{Pick\,Marker(s)}$', '$\mathrm{Navigation}$', '$\mathrm{Put\,Marker(s)}$']

    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, columnspacing=0.3, fontsize=11.5)

    plt.show()

if __name__ == '__main__':
    main()
