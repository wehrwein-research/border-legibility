import pandas as pd
import random

# Generates a csv file for a hit for turk
def make_batch(tiles_path, csv_path, batch_name, num_hits, imgs_src, num_pairs=30):
    values = []
    for i in range(num_pairs*2):
        values.append('item{}_image_url'.format(i+1))
    df = pd.DataFrame(columns=values, index=range(0, num_hits))
    #iterate through each line in the CSV - each row is a hit
    for i in range(num_hits):
        pairs = open(tiles_path).read().splitlines()
        line = []
        #get random pairs
        for j in range(num_pairs):
            check = True
            while check:
                randomline = random.choice(pairs)
                pair = randomline.split(',')
                if pair not in line:
                    check = False
            line.append(pair)
        row = []
        for item in line:
            for p in item:
                p = imgs_src + p[:-4]
                row.append(p)
        df_row = pd.Series(row, index=df.columns)
        df.iloc[i] = df_row
    path = '{}/{}.csv'.format(csv_path, batch_name)
    df.to_csv(path, index=False)
    return path
