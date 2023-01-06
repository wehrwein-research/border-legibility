import pandas as pd
import math

""" Calculates the various metrics used to evaluate the success
    of our methods.
    
    *Note: spearman metric uses pearson() method, using rankings instead of raw scores*
    
    Authors:
    Skyler Crane
    Trevor Ortega
"""

# For cases where our expirement is missing a tile,
# we remove that tile from ground truth
def make_rankings_same_size(r1, r2):
    if len(r1) != len(r2):
        print(len(r1), len(r2))
    r2_set = set(r2)
    r1 = [x for x in r1 if x in r2_set]
    return r1, r2

# Calculates Kendall's Tau by comparing ground truth rankings with the model rankings
def tau(gt_rank, new_rank):
    if len(gt_rank) != len(new_rank):
        gt_rank, new_rank = make_rankings_same_size(gt_rank, new_rank)
    distance = kendall_distance(gt_rank, new_rank)
    return (1)-((4*distance)/(len(gt_rank)*(len(gt_rank)-1)))

# Helper method for tau()
def kendall_distance(gt_rank, new_rank):
    if len(gt_rank) != len(new_rank):
        gt_rank, new_rank = make_rankings_same_size(gt_rank, new_rank)
    index_map = {rank:i for i, rank in enumerate(new_rank)}
    score = 0
    
    for i, rank in enumerate(gt_rank):
        for j, rank2 in enumerate(gt_rank):
            sigma_i = index_map[rank]
            sigma_j = index_map[rank2]
            
            if i < j and sigma_i > sigma_j:
                score += 1
    return score
                
# Calculates Spearman's Footrule by comparing ground truth rankings with the model rankings
def footrule(gt_rank, new_rank):
    if len(gt_rank) != len(new_rank):
        gt_rank, new_rank = make_rankings_same_size(gt_rank, new_rank)
    score=0  
    for i, rank in enumerate(gt_rank):
        j = new_rank.index(rank)
        score += abs(i-j)
    return score/len(gt_rank)

# Calculates the spearman/pearson correlation based on type of input.
# If gt and new are lists of the ranks, this will calculate the spearman correlation.
# If gt and new are lists of the raw model scores, this will calculate the pearson correlation.
def pearson(gt, new):
    if len(gt) != len(new):
        gt, new = make_rankings_same_size(gt, new)
    xy = []
    x2 = []
    y2 = []

    for i, ranks in enumerate(zip(gt, new)):
        xy.append(ranks[0]*ranks[1])
        x2.append(ranks[0]*ranks[0])
        y2.append(ranks[1]*ranks[1])

    n = len(gt)
    numerator = (n * sum(xy)) - (sum(gt)*sum(new))
    d1 = ((n * sum(x2)) - pow(sum(gt), 2))
    d2 = ((n * sum(y2)) - pow(sum(new), 2))
    denominator = math.sqrt(d1*d2)
    return numerator / denominator