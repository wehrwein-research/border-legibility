import sys, os, csv, math
from pathlib import Path
import warnings
warnings.simplefilter("ignore") # Change the filter in this process
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('border-legibility')+1])
print(str(PROJ_ROOT) + '/MODEL/GMM')
sys.path.append(str(PROJ_ROOT) + '/MODEL/GMM')
sys.path.append(str(PROJ_ROOT) + '/data')

from gmmy_bear import create_tri_mask
import border_utils as bu

sys.path.append('contrastive/mix')

if len(sys.argv) > 1:
    SCRAPE_DIR = str(PROJ_ROOT) + '/bing_maps/' + str(sys.argv[1])
else:
    SCRAPE_DIR = str(PROJ_ROOT) + '/bing_maps/global-scrape'

def run_mix_model(id=None, model=None):
    import random
    import numpy as np
    import torch 
    from contrastive.mix.model import SiameseDecider, SiameseDeciderOld
    from contrastive.mix.dataset import get_train_and_val_loader
    
    prefix = SCRAPE_DIR
    file = str(PROJ_ROOT) + "turk/experiments/annotations.csv"
    save_file = str(PROJ_ROOT) + '/MODEL/contrastive/mix/res/final4.ckpt'
    
    if model is None:
        weight_path = 'contrastive/mix/weights/final4.ckpt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SiameseDeciderOld.load_from_checkpoint(weight_path, 
                                                       batch_size=4, map_location=device)
        model = model.to(device)
    
    img_file = SCRAPE_DIR + '/tile-list.txt'
    train, val = get_train_and_val_loader(img_file, 10, 0.8, 1)
    def score_function(x1, x2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        name_1, name_2 = x1, x2
        reverse = random.random() > 0.5
        if reverse:
            x1, x2 = x2, x1
            name_1, name_2 = name_2, name_1
        og_x1, og_x2 = [bu.imread(x) for x in [x1, x2]]
        h, w, c = og_x1.shape
        mask_1, mask_2 = [create_tri_mask(x, (h,w), (h,w), color=2,
                               thick=5, show=False)
                     for x in [x1, x2]]
        x1 = train.dataset.prep_img(og_x1, rot=False).unsqueeze(0).to(device)
        x2 = train.dataset.prep_img(og_x2, rot=False).unsqueeze(0).to(device)
        y_hat = model(x1, x2).cpu().squeeze().softmax(dim=0).detach().tolist()
        y3 = 0 if len(y_hat) < 3 else y_hat[2]
        y_hat = y_hat[:2]
        if reverse:
            y_hat = y_hat[::-1]
        return y_hat + [y3]
    
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    return run(prefix, file, score_function, comparison_function, save_file=save_file)
    
    
def iter_get_prediction(root, pairs, score_func, compare_func, df, turk_mode=False):
    id_to_file = lambda idee: root + idee[:7] + "/" + idee + "/" + idee + ".jpeg"
    calc_df = pd.DataFrame()
    for i, tiles in enumerate(pairs):
        # Pre-compute scores if not already done
        tile_a, tile_b = tiles.split(',')
         
        if tiles not in calc_df:
            file1 = id_to_file(tile_a)
            file2 = id_to_file(tile_b)
            try:
                calc_df[tiles] = score_func(file1, file2)
            except Exception as e:
                raise e
                print(e)
                calc_df[tiles] = [-1, -1, -1]
                continue
        # Error tile, do nothing
        elif list(calc_df[tiles]) == [-1, -1, -1]:
            continue

        score_a, score_b, class3 = list(calc_df[tiles])
        # check if the choice for this file is the same as our choice
        choice = compare_func(tile_a, score_a, tile_b, score_b)
        match = 'yes' if choice == df['choice'][i] else 'no'
        
        if turk_mode:
            worker_id = df['worker_id'][i] if 'worker_id' in df.columns else '1'
            yield [tile_a, tile_b, score_a, score_b, match, worker_id]
        else:
            yield [tile_a, tile_b, score_a, score_b, match]


def run(root, annotations_file, score_func, compare_func, save_file=None):
    df = pd.read_csv(annotations_file)
    turk_mode = 'tile1' in df.columns
    if turk_mode:
        all_images = df['tile1'] + ',' + df['tile2']
        columns = ['tile1', 'tile2', 'score1', 'score2', 'match', 'worker_id']
        if 'choice' not in df.columns:
            df['choice'] = np.where(df['tile1count'] > df['tile2count'], 
                                    df['tile1'], df['tile2'])
    else:
        all_images = df.pop('id1') + ',' + df.pop('id2')
        columns = ['tile1', 'tile2', 'score1', 'score2', 'match']
        
    
    if save_file:
        with open(save_file, 'w+') as save_f:
            writer = csv.writer(save_f)
            writer.writerow(columns)
            for row in tqdm(iter_get_prediction(root, all_images,
                                                score_func, compare_func,
                                                df, turk_mode=turk_mode)):
                writer.writerow(row)
        res = pd.read_csv(save_file)
        acc = (res['match'] == 'yes').sum() / len(res)
        print('ACC:', acc)
        return acc
    else:
        for row in tqdm(iter_get_prediction(root, all_images, score_func,
                                            compare_func, df,
                                            turk_mode=turk_mode)):
            print(row)
            break


if __name__ == "__main__":
    run_mix_model()
