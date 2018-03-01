import os
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scatwave.scattering import Scattering
from dataloader import DataLoader
import matplotlib.pyplot as plt
import pickle

pca2 = PCA(n_components = 2)
features = None
with open('vggfeatures.file', 'rb') as file:
    features = pickle.load(file)

# PCA visualize
print 'doing PCA...'
X_r = pca2.fit(features).transform(features)
print 'PCA done!'

plt.figure()
colors = ['navy', 'turquoise']
target_names = np.array(['fake', 'real'])
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], color=color, alpha=.8, lw=0.2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of dataset')

# t-SNE visualize
print 'doing t-sne...'
X_tsne = TSNE(n_components=2).fit_transform(features)
print 't-sne done!'
plt.figure()
colors = ['navy', 'turquoise']
target_names = np.array(['fake', 'real'])
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], color=color, alpha=.8, lw=0.2,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('t-SNE of dataset')

plt.show()
