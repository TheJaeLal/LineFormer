import warnings
warnings.filterwarnings("ignore")

import metric6a

import infer

import mmcv
import os
from pathlib import Path
import time
import pandas as pd
import argparse
from tqdm import tqdm


def get_results(img_dir, annot_dir, post_proc):
    results = []
    for pname in tqdm(os.listdir(img_dir)):
        sample_name = Path(pname).stem
        img_path = img_dir + f"/{str(pname)}"
        annot_path = annot_dir + f"/{sample_name}.json"
        annot = mmcv.load(annot_path)
        charttype = annot['task1']['output']['chart_type']
        if 'line' != charttype.lower().strip():
            continue
        img = mmcv.imread(img_path)
        # print(annot['task6']['output']['visual elements']['lines'])
        try:
            pred_ds = infer.get_dataseries(img, annot=None, to_clean=False, post_proc=post_proc, mask_kp_sample_interval=10)
            # if sample_name == 'PMC6362862___7':
                # exit(0)
        except Exception as e:
            print('*'*8, f'Exception occured for: {img_path}', '*'*8)
            print('Exception:', e)
            raise
            pred_ds = []

        results.append({'name': sample_name, 'pred': pred_ds, 'gt': annot['task6']['output']['visual elements']['lines']})

    return results


def get_metric(results, score_func):
    s = []
    for sample in results:
        try:
            s.append({'name':sample['name'], 'score': score_func(sample['pred'], sample['gt'], gt_type="lines")})
        except:
            # https://github.com/scipy/scipy/pull/7031 Need this fix in scipy module.
            # only one case fialing, so ingoring.
            # print(edited_v)
            # print(annot['task6']['output']['visual elements']['lines'])
            print("Failed to caculate the score on " + sample['name'])
            pass
    s = pd.DataFrame(s)
    return s

def handle_arg_errors(args):
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"{args.data_dir} does not exist!")
    elif not Path(f"{args.data_dir}/images/").exists():
        raise FileNotFoundError(f"Image Directory {args.img_dir} does not exist!")
    elif not Path(f"{args.data_dir}/annot/").exists():
        raise FileNotFoundError(f"Annotation Directory {args.annot_dir} does not exist!")
    elif not Path(args.model_config).exists():
        raise FileNotFoundError(f"Model config path {args.model_config} does not exist!")
    elif not Path(args.model_ckpt).exists():
        raise FileNotFoundError(f"Model ckpt {args.model_ckpt} does not exist!")

def main():

    parser = argparse.ArgumentParser(description='Process some data.')
    
    # Add arguments
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('--model_config', type=str, nargs='?', default="lineformer_swin_t_config.py", help='Path to model config')
    parser.add_argument('--model_ckpt', type=str, nargs='?', default="iter_3000.pth", help='Path to saved model checkpoint')
    parser.add_argument('--device', type=str, nargs='?', default="cuda:0", help='Device to run model inference')
    parser.add_argument('--postproc', action='store_true', help='Turn on postprocessing in data extraction')

    # Parse arguments
    args = parser.parse_args()

    handle_arg_errors(args)

    # Load the model
    infer.load_model(args.model_config, args.model_ckpt, args.device)

    # Access the arguments
    annot_dir = f"{args.data_dir}/annot/"
    img_dir = f"{args.data_dir}/images/"

    # Run Inference on all the samples...
    print('Evaluating on :', args.data_dir)
    results = get_results(img_dir, annot_dir, post_proc=args.postproc)

    print('Calculating 6a score')
    df_6a = get_metric(results, score_func=metric6a.metric_6a_indv)
    print('Calculating 6b score')
    df_6b = get_metric(results, score_func=metric6a.metric_6b_indv)

    print('------------Results------------')
    print("Average 6a score: "+str(df_6a['score'].mean()))
    print("Average 6b score: "+str(df_6b['score'].mean()))

    df_scores = pd.merge(df_6a, df_6b, how='outer', on='name', suffixes=("_6a", "_6b"))
    fname = f"results_{time.time()}.csv"
    
    df_scores.to_csv(fname, index=False)
    print('******Saving scores as:', fname, "*******")

    return

if __name__ == '__main__':
    main()
