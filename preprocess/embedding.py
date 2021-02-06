import torch
import musdb




def extract_vggish(dataset, fs=44100):
	model = torch.hub.load('harritaylor/torchvggish', 'vggish')
	model.eval()

	for track in dataset:
		vgg_feature = model.forward(track.audio, fs)
		print(vgg_feature.shape)


if __name__ == '__main__':
	root = '../../data/MUSDB18-HQ/'
	mus_train = musdb.DB(root=root, is_wav=True, split='train', subsets='train', download=False)
	mus_valid = musdb.DB(root=root, is_wav=True, split='train', subsets='valid', download=False)
	mus_test = musdb.DB(root=root, is_wav=True, split='test', subsets='test', download=False)

	extract_vggish(mus_test)

