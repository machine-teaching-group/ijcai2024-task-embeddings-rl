import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from envs.CartPoleVar import CartPoleVarEnv
import numpy as np
import torch
from models import EmbeddingNetwork
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

plt.style.use('science')

EMBEDDING_NET_PATH = 'vis_embedding_scripts/embedding_networks/embedding_model_withNormConstraints_CartPoleVar_dim3.pt'

embedding_dim = 3

def get_attributes(task, env):
    t_type = int(task[env.key2idx["task_type"]])
    t_sign = int(task[env.key2idx["force_magnitude"]] >= 0)
    
    shape = 'o'
    color = 'red' if (t_type != t_sign) else 'blue'
    
    return (color, shape)  

def main():
    env = CartPoleVarEnv()
    
    model_data = torch.load(EMBEDDING_NET_PATH, map_location=torch.device('cpu'))
    embedding_model = EmbeddingNetwork('CartPoleVar', embedding_dim)
    embedding_model.load_state_dict(model_data['parameters'])
    embedding_model.eval()  
    
    random.seed(0)
    np.random.seed(100) 
    
    tasks = [env.reset() for _ in range(10000)]
    tasks = [(task, *get_attributes(task, env)) for task in tasks]

    embeddings = np.zeros((len(tasks), embedding_dim))
    
    for i, (x, _, _) in enumerate(tasks):
        x_b = torch.FloatTensor(x).view(1, -1)
        with torch.no_grad():
            embedding = embedding_model(x_b)
            embeddings[i] = embedding   
    
    norms = np.linalg.norm(embeddings, axis=1)        
    tsne_embeddings = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=5).fit_transform(embeddings)          
    
    for i, (x, color, marker) in enumerate(tasks):
        embedding = tsne_embeddings[i]
    
        plt.scatter(embedding[0], embedding[1], c=(color), marker=marker, s=norms[i]*10, alpha=0.3, rasterized=True, edgecolors='black')
        
    colors = ['red', 'blue']
    markers=["o"]
    
    plt.xticks([])
    plt.yticks([])
    
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none", alpha=0.6)[0]
    
    handles = [f("o", colors[i]) for i in range(2)]
    handles += [f(markers[i], "k") for i in range(1)]

    labels = ['$\mathrm{Action\,0:\,Left}$', '$\mathrm{Action\,0:\,Right}$']

    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, columnspacing=0.8, fontsize=11.5) 

    plt.show()

if __name__ == '__main__':
    main()
