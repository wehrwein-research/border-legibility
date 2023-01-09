import sys, os, csv, math
import warnings
warnings.simplefilter('ignore')

import pandas as pd
from tqdm import tqdm

sys.path.append('GMM')
sys.path.append('../data')

import border_utils as bu

'''
This file contains all the code needed to run the baseline methods from the paper.
To run a method: 
1) In the main method, uncomment the method you wish to run.
1.5) If necessary, change method parameters. Options: (layer=1,2,3), (whole=True,False).
2) Ensure that paths at top of main method are updated.
3) Run file
'''

def run_gaussian_pyramid(prefix, file):
    sys.path.append("baselines")
    import baseline

    save_file = f"{SAVE_DIRECTORY}/gradient_pyramid.csv"
    
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y
    
    run(prefix, file, baseline.pyramid_gradient, comparison_function, save_file)

def run_gradient_norm(prefix, file):
    sys.path.append("baselines")
    import baseline
    
    save_file = f'{SAVE_DIRECTORY}/gradient_norm.csv'
    
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y
    
    run(prefix, file, baseline.norm_of_gradients, comparison_function, save_file)
        
def run_deep_cluster_entropy(prefix, file,layer=1, whole=True):
    from deep import prep_model, arr2model, distribution_kmeans_preds
    from gmmy_bear import create_tri_mask
    from scipy.stats import wasserstein_distance as emd
    from scipy.special import kl_div
    
    if whole == True:
        save_file = f'{SAVE_DIRECTORY}/deep-cluster-entropy-kl-WHOLE-layer{str(layer)}.csv'
    else:
        save_file = f'{SAVE_DIRECTORY}/deep-cluster-entropy-kl-layer{str(layer)}.csv'
    
    kl = lambda x, y: kl_div(x, y).mean()
    model, transform = prep_model(layer=layer)
    m = lambda x: arr2model(x, model, transform).permute((1,2,0))
    
    # Sets mask properties based on layer
    i = 56
    if layer == 2:
        i = 28
    elif layer == 3:
        i = 14
            
    def score_function(img):
        h, w, c = bu.imread(img).shape
            
#         deep_tri_mask = create_tri_mask(img, (h,w), (i,i), color=2, thick=2, show=False, )
        deep_tri_mask = create_tri_mask(img, (h,w), (i,i), color=2, thick=2, show=False,one_hot=False)

        return distribution_kmeans_preds(img, deep_tri_mask, kl, n_clusters=3, model=m, vis=False, whole_to_side=whole)
    
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    run(prefix, file, score_function, comparison_function, save_file)


def run_mae_cluster_entropy(prefix, file):
    sys.path.append('maskautoencode/mae')
    import maskautoencode.mae.borderMAEUtils as bmu
    from scipy.special import kl_div

    save_file = f'{SAVE_DIRECTORY}/mae-cluster-entropy.csv'
    
    distr_func = lambda x, y: kl_div(x, y).mean()
    # maps images to scores
    score_function = lambda img: bmu.distribution_kmeans_preds(img, distr_func, vis=False) 
    
    # Return the winner given the two scores
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    run(prefix, file, score_function, comparison_function, save_file)   

def run_pixel_cluster_entropy(prefix, file):
    from deep import prep_model, arr2model, distribution_kmeans_preds
    from gmmy_bear import create_tri_mask
    from scipy.stats import wasserstein_distance as emd
    from scipy.special import kl_div
    
    save_file = f'{SAVE_DIRECTORY}/pixel-cluster-entropy.csv'
    
    kl = lambda x, y: kl_div(x, y).mean()
    model, transform = prep_model()
    m = lambda x: arr2model(x, lambda x: x, transform).permute((1,2,0))
    def score_function(img):
        h, w, c = bu.imread(img).shape
        deep_tri_mask = create_tri_mask(img, (h,w), (224,224), color=2, thick=5, show=False)
        return distribution_kmeans_preds(img, deep_tri_mask, kl, n_clusters=3, model=m, vis=False)
    
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    run(prefix, file, score_function, comparison_function, save_file)

def run_pixel_similarity(prefix, file):
    from deep import run_func_on_sides, similarity_gambit_2d_mask, prep_model, similarity_correspondance_2d_mask
    
    save_file = f'{SAVE_DIRECTORY}/pixel-cosine.csv'
    
    model1, transform = prep_model()
    score_function = lambda img: run_func_on_sides(img, similarity_gambit_2d_mask, 
                                                   model1, transform, vis=False)
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    
    run(prefix, file, score_function, comparison_function, save_file)

def run_deep_feature_similarity(prefix, file, layer=1):
    from deep import run_func_on_sides, similarity_gambit_2d_mask, prep_model, similarity_correspondance_2d_mask
    
    save_file = f'{SAVE_DIRECTORY}/deep-cosine-L2-layer{str(layer)}.csv'
    
    model, transform = prep_model(layer=layer)
    score_function = lambda img: run_func_on_sides(img, similarity_gambit_2d_mask, 
                                                   model, transform, vis=False)
    #score_function = lambda img: run_func_on_sides(img, similarity_correspondance_2d_mask, 
    #                                               model1, transform, vis=False)
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    
    run(prefix, file, score_function, comparison_function, save_file)
    
def run_mae_gmm_distance(prefix, file):
    sys.path.append('maskautoencode/mae')
    import maskautoencode.mae.borderMAEUtils as bmu
    
    save_file = f'{SAVE_DIRECTORY}/mae-gmm-distance.csv'
    
    # maps images to scores
    score_function = lambda img: bmu.cosine_distance_gmm_img(img, vis=False)
    
    # Return the winner given the two scores
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 
    run(prefix, file, score_function, comparison_function, save_file)

def run_feature_classifier(prefix, file, bal='normal', train_size=80, test_size=20, rotate=False):
    sys.path.append('classify')
    import classification as clf
    
    save_file = f'{SAVE_DIRECTORY}/feature-classifier.csv'
    
    # maps images to scores
    score_function = lambda img: clf.run_triple_classification(img, train_split=train_size, test_split=test_size, rot=rotate)[0] 
    
    # Given two scores and their tile names, return the winner's name
    comparison_function = lambda name_x, x, name_y, y: name_x if x > y else name_y 

    # this will run your method and save the outputs to the save file
    run(prefix, file, score_function, comparison_function, save_file)

def iter_get_prediction(root, tile_As, tile_Bs, score_func, compare_func, calc_df, df, is_turk):
    id_to_file = lambda idee: root + idee[:7] + "/" + idee + "/" + idee + ".jpeg"
    num_errs = 0
    for i, (tile_a, tile_b) in enumerate(zip(tile_As, tile_Bs)):
        err = False
        for f in [tile_a, tile_b]:
            # Pre-compute scores if not already done
            if math.isnan(calc_df[f]):
                file = id_to_file(f)
                try:
                    calc_df[f] = score_func(file)
                    if calc_df[f] == float('inf'):
                        break
                except Exception as e:
                    err = True
                    print(e)
                    calc_df[f] = float('inf')
                    break
            # Error tile, do nothing
            elif calc_df[f] == float('inf'):
                err = True
                break

        if err:
            num_errs += 1
            continue

        score_a, score_b = calc_df[tile_a], calc_df[tile_b]


        # check if the choice for this file is the same as our choice
        
        choice = compare_func(tile_a, score_a, tile_b, score_b)
        
        if is_turk:
            try:
                annotation = tile_As[i] if df['tile1count'][i] > df['tile2count'][i] else tile_Bs[i]
                match = 'yes' if choice == annotation else 'no' 
    #             uncertainty = all(df.iloc[[i]][key] > 0 for key in ['tile1count', 'tile2count'])
                uncertainty = False
                yield [tile_a, tile_b, score_a, score_b, match, uncertainty]
            except KeyError:
                annotation = df['choice'][i]
                match = 'yes' if choice == annotation else 'no' 
                uncertainty = False
                worker_id = df['worker_id'][i]
                yield [tile_a, tile_b, score_a, score_b, match, uncertainty, worker_id]
        else:
            match = 'yes' if choice == df['choice'][i] else 'no'
            yield [tile_a, tile_b, score_a, score_b, match]

def run(root, annotations_file, score_func, compare_func, save_file=None):
    df = pd.read_csv(annotations_file)
    
    turk_mode = 'tile1' in df.columns
    if turk_mode:
        tile_As = df.pop("tile1")
        tile_Bs = df.pop("tile2")
        columns = ['tile1', 'tile2', 'score1', 'score2', 'match', 'uncertain']
        raw_annotations = 'worker_id' in df.columns
        if raw_annotations:
            columns.append('worker_id')
    else:
        tile_As = df.pop("id1")
        tile_Bs = df.pop("id2")
        columns = ['tile1', 'tile2', 'score1', 'score2', 'match']
    
    all_images = tile_As.append(tile_Bs).unique()
    id_to_file = lambda idee: root + idee[:7] + "/" + idee + "/" + idee + ".jpeg"
    tot_err = 0

    # Dataframe to hold pre-computed calculations
    calc_df = pd.DataFrame(data=[float('nan')]*len(all_images), index=all_images)[0]
    
    if save_file:
        with open(save_file, 'w+') as save_f:
            writer = csv.writer(save_f)
            writer.writerow(columns)
            for row in tqdm(iter_get_prediction(root, tile_As, tile_Bs, score_func, compare_func, calc_df, df, turk_mode)):
                writer.writerow(row)
        df = pd.read_csv(save_file)
        accuracy = (df['match'] == 'yes').sum() / len(df)

            
        import pathlib as path
        sys.path.append('pairwise')
        import rank as rank
        import metrics as metrics
        method_name = path.Path(save_file).stem
        turk_save_path = 'pairwise/results/turk_pilot.csv' #ranked annotations
        print(df)
        model_save_path = 'pairwise/results/model_rankings/'+ method_name + '_ranked.csv' #ranked model choices
        
        rank.analyze_turk_csv(annotations_file, turk_save_path, raw=False) 
        rank.analyze_modelComparisons_csv(save_file, model_save_path)

        df1 = pd.read_csv(turk_save_path)
        df2 = pd.read_csv(model_save_path)

        id_map = {x: i for i, x in enumerate(all_images, 1)}
        # takes in rank, and returns score associated with that rank
        # will produce error if the x is not in the df['name']
        def id_map_score(x, df):
            row = df.index[df.reset_index()['name']==x].tolist()[0]
            return df.loc[row, 'value']

        # removes any row from df1 where 'name' does not exist in df2
        def match_df(df1, df2):
            for name in df1['name']:
                try:
                    row = df2.index[df2.reset_index()['name']==name].tolist()[0]
                except:
                    row = df1.index[df1.reset_index()['name']==name].tolist()[0]
                    df1.drop(index=row, inplace=True)
            return df1, df2

        df1 = df1.sort_values('value', ascending=False)   
        df2 = df2.sort_values('value', ascending=False)

        gt_rank = list(df1['name'].apply(lambda x: id_map[x]))
        rank = list(df2['name'].apply(lambda x: id_map[x]))

        df1, df2 = match_df(df1, df2)

        gt_rank_score = list(df1['name'].apply(lambda x: id_map_score(x,df1)))   
        rank_score = list(df1['name'].apply(lambda x: id_map_score(x,df2)))

    #     run metrics
        tau = metrics.tau(gt_rank[::-1], rank)
        footrule = metrics.footrule(gt_rank[::-1], rank)
        kendall = metrics.kendall_distance(gt_rank[::-1], rank)
        spearman = metrics.pearson(gt_rank[::-1], rank) #spearman is pearson on rankings instead of raw scores
        pearson = metrics.pearson(gt_rank_score[::-1], rank_score)


        df3 = pd.read_csv(METRICS_RESULTS)
        if method_name in df3["method"].values:
            i = df3.loc[df3["method"] == method_name].index[0]
            df3.loc[i:i, 'method':'pearson'] = method_name, accuracy, tau, footrule, kendall, spearman, pearson
        else:
            row = pd.DataFrame([[method_name, accuracy, tau, footrule, kendall, spearman, pearson]], columns=["method", "accuracy","tau", "footrule", "kendall", "spearman", "pearson"])
            pd.concat([df3, row], ignore_index=True)
            df3 = df3.append(row, ignore_index=True)
        df3.to_csv(METRICS_RESULTS, index=False)

    else:
        for row in tqdm(iter_get_prediction(root, tile_As, tile_Bs, score_func, compare_func, calc_df, df, turk_mode)):
            print(row)
            
    
            
if __name__ == "__main__":
    
    # path to save metric results to
    METRICS_RESULTS = 'pairwise/results/metrics/metrics.csv'
    
    # directory to save model results
    SAVE_DIRECTORY = 'pairwise/results/benchmarks'
    
    # directory of scrape
    prefix = "../bing_maps/global_scrape/"
    
    # path to annotations file
    annotations_file = "../turk/experiments/annotations.csv"
    
###################
##### METHODS #####
###################

##### From main paper, as labeled in Table 1

    ## Clustering:
    # Pixel
#     run_pixel_cluster_entropy(prefix, annotations_file)
    # Convolutional Feature Layers
#     run_deep_cluster_entropy(prefix, annotations_file, layer=3, whole=True)
    # MAE Features
   # run_mae_cluster_entropy(prefix, annotations_file)

    ## Distance
    # Pixel
#     run_pixel_similarity(prefix, annotations_file)
    # Convolutional Feature Layers
#     run_deep_feature_similarity(prefix, annotations_file, layer=3)   
    
##### From Supplemental section

    ## Distance GMM
    # MAE Features
#     run_mae_gmm_distance(prefix, annotations_file)
    
#### Bonus
#     run_gradient_norm(prefix, annotations_file)
#     run_gaussian_pyramid(prefix, annotations_file)
#     run_feature_classifier(prefix, annotations_file)
