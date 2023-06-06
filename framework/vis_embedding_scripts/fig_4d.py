import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import torch
from models import EmbeddingNetwork
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

plt.style.use('science')

EMBEDDING_NET_PATH = 'vis_embedding_scripts/embedding_networks/embedding_model_withoutNormConstraints_MultiKeyNavA_dim3.pt'
embedding_dim = 3

def main():
    model_data = torch.load(EMBEDDING_NET_PATH, map_location=torch.device('cpu'))
    embedding_model = EmbeddingNetwork("MultiKeyNavA", embedding_dim)
    embedding_model.load_state_dict(model_data['parameters'])
    embedding_model.eval()
    
    markers = {'00': 'D', '01': '*', '10':'P', '11':"o"}
    
    tasks = [([i, j, k, l, m, n, o], 'red', markers[f'{n}{o}']) for i in [0.05, 0.45, 0.85] for j in range(1) for k in range(1) for l in range(1) for m in range(1) for n in range(2) for o in range(2)]
    tasks += [([i, j, k, l, m, n, o], 'blue', markers[f'{n}{o}']) for i in [0.05, 0.45, 0.85] for j in range(1, 2) for k in range(1) for l in range(1) for m in range(1) for n in range(2) for o in range(2)]
    tasks += [([i, j, k, l, m, n, o], 'green', markers[f'{n}{o}']) for i in [0.05, 0.45, 0.85] for j in range(1) for k in range(1, 2) for l in range(1) for m in range(1) for n in range(2) for o in range(2)]
    tasks += [([i, j, k, l, m, n, o], 'yellow', markers[f'{n}{o}']) for i in [0.05, 0.45, 0.85] for j in range(1) for k in range(1) for l in range(1, 2) for m in range(1) for n in range(2) for o in range(2)]
    tasks += [([i, j, k, l, m, n, o], 'orange', markers[f'{n}{o}']) for i in [0.05, 0.45, 0.85] for j in range(1) for k in range(1) for l in range(1) for m in range(1, 2) for n in range(2) for o in range(2)]
    tasks += [([i, j, k, l, m, n, o], 'black', markers[f'{n}{o}']) for i in [0.05, 0.45, 0.85] for j in range(1, 2) for k in range(1, 2) for l in range(1, 2) for m in range(1, 2) for n in range(2) for o in range(2)]
    
    tasks += [(x, 'black', markers[f'{x[-2]}{x[-1]}']) for x in [[0.95,1,1,1,1,0,0], [0.95,1,1,1,1,0,1], [0.95,1,1,1,1,1,0], [0.95,1,1,1,1,1,1]]]
    embeddings = np.zeros((len(tasks), embedding_dim))
    
    for i, (x, _, _) in enumerate(tasks):
        x_b = torch.FloatTensor(x).view(1, -1)
        with torch.no_grad():
            embedding = embedding_model(x_b)
            embeddings[i] = embedding
           
    np.random.seed(1000)
    tsne_embeddings = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embeddings)  
    
    for i, (x, color, marker) in enumerate(tasks):
        embedding = tsne_embeddings[i]
    
        plt.scatter(embedding[0], embedding[1], c=(color), marker=marker, alpha=0.4, edgecolors='black')
        
    plt.annotate('(Pick A)', (-4.1, -1.19), c=('maroon'))
    plt.annotate('(Navigation)', (0.6, -7.2), c=('maroon'))
    
    plt.xticks([])
    plt.yticks([])

    plt.show()

if __name__ == '__main__':
    main()