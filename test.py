from separator import Separator
import argparse
import glob
import os
import museval
import tqdm

def main(args):
    model = Separator.load_from_checkpoint(glob.glob(args.ckpt_path+"/*.pth")[0], default_params=glob.glob(args.ckpt_path+"/*.yaml")[0])

    results = museval.EvalStore()

    test_set = model._get_data_loader('test')
    for mix_audio, tar_audio, track_id in tqdm.tqdm(test_set):
    	score = model.test_step(mix_audio, tar_audio, track_id)
    	results.add_track(score)
    
    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, model.hparams.model_name)
    method.save(args.ckpt_path + '.pandas')

    

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="")

	parser.add_argument('--ckpt_path', type=str, default="logger/default/version_10", help='path to check point, only useful when model equal tcn or vgg')

	args = parser.parse_args()

	main(args)