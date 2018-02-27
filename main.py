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
from converter import send_crops_of_this_dispute_painting
import matplotlib.pyplot as plt
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# params
iters = 300
batch_size = 10
shapeofoutput = 245196
features = np.zeros((iters * batch_size, shapeofoutput), dtype=np.float32)
labels = np.zeros(iters * batch_size, dtype=np.uint8)
scat = Scattering(M=224, N=224, J=4).cuda()
loader = DataLoader()
pca2 = PCA(n_components = 2)

# scatter feature
for i in range(iters):
    print 'Iter: %s/%s' % (i, iters)
    batch, label_batch = loader.get_batch(10, 'train', 'rgb')
    for j in range(batch.shape[0]):
        x = batch[j][np.newaxis,:]
        label = label_batch[j]
        x = x.transpose(0,3,1,2)
        x = torch.from_numpy(x).float().cuda() #numpy2tensor
        output = scat(x)
        output = output.view(-1)
        output = output.cpu().numpy()
        #output = output[np.newaxis,:]
        #output = pca4096.fit(output).transform(output)
        features[i*batch_size + j] = output
        labels[i*batch_size + j] = label

# svm training
print 'Training SVM...'
clf = svm.SVC(probability=True)
clf.fit(features, labels)
print 'SVM done!'

# svm prediction
DISPUTE = [1, 7, 10, 20, 23, 25, 26]
for i in range(len(DISPUTE)):
    testdata = send_crops_of_this_dispute_painting(DISPUTE[i])
    testdata = testdata.transpose(0,3,1,2)
    print("painting number ",DISPUTE[i])
    allfeatures_thispainting = []
    for j in range(testdata.shape[0]):
        features_onecrop = scat(torch.from_numpy(testdata[j:j+1,:]).float().cuda())
        features_onecrop = features_onecrop.view(-1).cpu().numpy()
        allfeatures_thispainting.append(features_onecrop)
    pred = clf.predict_proba(np.squeeze(np.array(allfeatures_thispainting)))
    scores = np.sum(pred,axis=0)
    print(scores)
    if scores[0] > scores[1]:
        print("Painting number ", DISPUTE[i] ," is fake")
    else:
        print("Painting number ", DISPUTE[i], " is real")
# print 'Saving SVM model...'
# with open('model.file', 'wb') as file:
#     pickle.dump(clf, file)
# print 'Model saved!'

# print 'saving features and labels...'
# with open('feature.file', 'wb') as file:
#     pickle.dump(features, file)
# with open('label.file', 'wb') as file:
#     pickle.dump(labels, file)
# print 'features and label saved!'
# # PCA visualize
# print 'doing PCA...'
# X_r = pca2.fit(features).transform(features)
# print 'PCA done!'
#
# plt.figure()
# colors = ['navy', 'turquoise']
# target_names = np.array(['fake', 'real'])
# for color, i, target_name in zip(colors, [0, 1], target_names):
#     plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], color=color, alpha=.8, lw=0.2,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('PCA of dataset')
#
# # t-SNE visualize
# print 'doing t-sne...'
# X_tsne = TSNE(n_components=2).fit_transform(features)
# print 't-sne done!'
# plt.figure()
# colors = ['navy', 'turquoise']
# target_names = np.array(['fake', 'real'])
# for color, i, target_name in zip(colors, [0, 1], target_names):
#     plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], color=color, alpha=.8, lw=0.2,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('t-SNE of dataset')
#
# plt.show()
