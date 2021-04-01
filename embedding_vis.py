import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

drums = []
bass = []
vocals = []
others = []

for file in os.listdir('./tmp/'):
	if file.split('_')[0] == '0':
		vocals.append(np.load('./tmp/'+file))
	if file.split('_')[0] == '1':
		drums.append(np.load('./tmp/'+file))
	if file.split('_')[0] == '2':
		bass.append(np.load('./tmp/'+file))
	if file.split('_')[0] == '3':
		others.append(np.load('./tmp/'+file))

size = 10000000
drums = np.concatenate(drums, 0)[:size]
bass = np.concatenate(bass, 0)[:size]
vocals = np.concatenate(vocals, 0)[:size]
others = np.concatenate(others, 0)[:size]
length = drums.shape[0]
X = np.concatenate((drums, bass, vocals, others))
print(drums.shape, bass.shape, vocals.shape)

tsne = TSNE(n_components=2, random_state=0)
output = tsne.fit_transform(X)
print(output.shape)

plt.scatter(output[:length, 0], output[:length, 1], c='r', label='0')
plt.scatter(output[length:length*2, 0], output[length:length*2, 1], c='g', label='1')
plt.scatter(output[length*2:length*3, 0], output[length*2:length*3, 1], c='b', label='2')
plt.scatter(output[length*3:length*4, 0], output[length*3:length*4, 1], c='y', label='3')

plt.savefig('vgg_emd.png')
