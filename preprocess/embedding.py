import torch
import musdb
import numpy as np
import tqdm
import sys


def extract_deep_salience(dataset):
	sys.path.append('../../ismir2017-deepsalience/predict/')
	from predict_on_audio import compute_features

	for track in tqdm.tqdm(dataset):
		emb = compute_features(track.audio).T
		np.save('../../data/MUSDB18-HQ/salience/'+track.name, emb)
		print(track.name, emb.shape)

def extract_sidd_att(dataset):
	sys.path.append('../../AttentionMIC/')
	from prediction import prediction

	for track in tqdm.tqdm(dataset):
		print(track)
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
	root = '../../data/MUSDB18-HQ/'
	mus_train = musdb.DB(root=root, is_wav=True, split='train', subsets='train', download=False)
	mus_valid = musdb.DB(root=root, is_wav=True, split='valid', subsets='train', download=False)
	mus_test = musdb.DB(root=root, is_wav=True, split='test', subsets='test', download=False)

	extract_deep_salience(mus_test)
	extract_deep_salience(mus_valid)
	extract_deep_salience(mus_train)

