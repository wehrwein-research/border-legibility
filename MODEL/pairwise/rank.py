import stan
import numpy as np
import pandas as pd

'''
args:
    data (pd.DataFrame): annotation data with columns 'id1', 'id2', 'choice', 'workerId'
        the choice field should be one of 'id1' or 'id2' and workerId should be an int
'''
def analyze_annotations(df, save_file=None, chains=3, n_iter=2500, seed=1234, n_cores=3):
    random_seed = 1
    model_code = '''
        data {
        int N; // number of comparisons
        int M; // number of documents
        int P; //Number of coders
        int y[N]; // outcome
        int g[N];    // id map first item in comparison
        int h[N];    // id map of second item in comparison
        int j[N]; // id map for workers
        }
        parameters {
        real a[M];
        real<lower=0> b[P];
        real<lower=0> sigma;  
        }
        model {
        sigma~normal(0,3);
        for(p in 1:P){
        b[p] ~ normal(0,sigma);
        }
        for(m in 1:M){
        a[m] ~ normal(0,1);
        }
        for(n in 1:N) {
        y[n] ~ bernoulli(inv_logit(b[j[n]]*(a[g[n]]-a[h[n]])));
        }
        }
    '''
    
    id1s = df.pop("id1")
    id2s = df.pop("id2")
    choice = df.pop("choice")
    
    all_ids = id1s.append(id2s).unique()
    id_map = {x: i for i, x in enumerate(all_ids, 1)}
    
    all_workers = df.pop("workerId")
    unique_workers = all_workers.unique()
    worker_map = {x: i for i, x in enumerate(unique_workers, 1)}
    
    g = np.array(id1s.apply(lambda x: id_map[x])).astype(int)
    h = np.array(id2s.apply(lambda x: id_map[x])).astype(int)
    j = np.array(all_workers.apply(lambda x: worker_map[x]))
    
    N = len(df)
    M = len(all_ids)
    P = len(unique_workers)
    y = np.array(choice == id2s, dtype=int)
    
    data = {
        "y":y, "g":g, "h":h, "N":N, "M":M, "P":P, "j":j
    }
    posterior = stan.build(model_code, data=data, random_seed=random_seed)
    
    fit = posterior.sample(num_chains=chains, num_samples=n_iter)
    
    df = fit.to_frame()
    pd.options.display.max_columns = len(df.columns)
    means = df.describe().T['mean']
    
    id_map = {i: x for i, x in enumerate(all_ids, 1)}
    # There exists a column of format a.i for each i from 1 to N
    probs = [means[i] for i in df.columns if 'a.' in i]
    results = {id_map[i]: x for i, x in enumerate(probs, 1)}
    
    worker_scores = [means[i] for i in df.columns if 'b.' in i]
    worker_map = {i: x for i, x in enumerate(unique_workers, 1)}
    worker_results = {worker_map[i]: x for i, x in 
                          enumerate(worker_scores, 1)}
    
    workers = sorted([(x,y) for x, y in worker_results.items()], 
            key=lambda w: w[1])
    for x, y in workers:
        print(x, y)
    
    
    if save_file is not None:
        results_df = pd.DataFrame({
            'name': list(results.keys()),
            'value': list(results.values())
        })
        sort_df = results_df.sort_values('value', ascending=False)
        rank = np.arange(len(sort_df)) + 1
        sort_df['rank'] = rank
        sort_df.to_csv(save_file, columns=['name', 'value', 'rank'])

    return results
    
'''
Model Comparisons file columns: tile1, tile2, choice, match

choice: filename matching either tile1 or tile2
match: either a 'yes' or 'no'. 'yes' if choice == tile1 else 'no'
'''
def analyze_modelComparisons_csv(csv_path, save_path, data_frame=None):
    if data_frame is None: #want to pass in a dataframe rather than a csv in certain situations
        df = pd.read_csv(csv_path)
    else:
        df = data_frame
    id1s = df.pop("tile1")
    id2s = df.pop("tile2")
    match = df.pop("match")

    df['choice'] = np.where(match == 'yes', id1s, id2s)
    df['id1'] = id1s
    df['id2'] = id2s
    df['workerId'] = np.ones(df.shape[0])
    sumin = analyze_annotations(df, save_file=save_path)

'''
Turk raw annotation columns of interest:
    tile1,tile2,choice,worker_id,batch_name,hit_id,assignment_id,submit_time
    
choice: one of tile1 or tile2
'''
def analyze_turk_csv(csv_path, save_path, raw=False):
    df = pd.read_csv(csv_path)
    id1s = df.pop("tile1")
    id2s = df.pop("tile2")
    if raw:
        df['workerId'] = df.pop("worker_id")
    else:   
        tile1count = df.pop("tile1count")
        tile2count = df.pop("tile2count")
        df['choice'] = np.where(tile1count > tile2count, id1s, id2s)
        df['workerId'] = np.ones(df.shape[0])
    df['id1'] = id1s
    df['id2'] = id2s
    sumin = analyze_annotations(df, save_file=save_path)