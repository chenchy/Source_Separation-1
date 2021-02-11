import torch
import musdb
import numpy as np
import tqdm
import sys


def extract_sidd_att(dataset):
	sys.path.append('../AttentionMIC/')
	from prediction import prediction

	for track in tqdm.tqdm(dataset):
		emb = prediction(np.load(f'../data/MUSDB18-HQ/vggish/{track.name}.npy')).T
		np.save('../data/MUSDB18-HQ/sidd_att/'+track.name, emb)
		print(track.name, emb.shape)
		# frames, features


def extract_vggish(dataset, fs=44100):
	model = torch.hub.load('harritaylor/torchvggish', 'vggish')
	model.eval()

	for track in tqdm.tqdm(dataset):
		vgg_feature = model.forward(track.audio, fs)
		np.save('../../data/MUSDB18-HQ/vggish/'+track.name, vgg_feature.cpu().detach().numpy())
		print(track.name, vgg_feature.shape)


if __name__ == '__main__':
	root = '../data/MUSDB18-HQ/'
	mus_train = musdb.DB(root=root, is_wav=True, split='train', subsets='train', download=False)
	mus_valid = musdb.DB(root=root, is_wav=True, split='valid', subsets='train', download=False)
	mus_test = musdb.DB(root=root, is_wav=True, split='test', subsets='test', download=False)

	extract_sidd_att(mus_test)
	extract_sidd_att(mus_valid)
	extract_sidd_att(mus_train)

