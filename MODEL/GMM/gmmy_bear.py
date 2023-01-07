import os, sys, random, time
from pathlib import Path, PosixPath
from itertools import permutations, combinations

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('border-legibility')+1])
sys.path.append(str(PROJ_ROOT) + '/data')
import border_utils as bu

import numpy as np
import imageio
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import cv2
from sultan.api import Sultan as Bash

class GMM(GaussianMixture):
    def __init__(self, n_components, random_state, value):
        super().__init__(n_components=n_components, random_state=random_state)
        self.value = value

def show_samples(gmms, names, dims=(20, 20)):
    f, axarr = plt.subplots(1, 3, sharey=True)
    print('Random samples from learned gaussian distribution:')
    h, w = dims
    i = 0
    for gm, name in zip(gmms, names):
        x, labels = gm.sample(h*w)
        axarr[i].imshow(x.reshape(h, w, 3).astype(np.uint8))
        axarr[i].set_title(name)
        i += 1

def create_tri_mask(tile, old_dims, new_dims, color=2, thick=5, show=False, one_hot=False):
    lines = None
    if type(tile) in [PosixPath, str]:
        tile = str(Path(tile).with_suffix('.npy'))
        try:
            lines = np.load(tile)
            if lines.shape[0] < 1:
                return None
        except Exception as e:
            print(e)
            return
    else:
        lines = tile
    lines = bu.resize_segments(lines, old_dims, new_dims)
    mask = bu.mask_from_segments(lines, dims=new_dims, draw_lines=True,
                                 color=color, thick=thick)
    # side1 = 0 (red), side2 = 1 (green), side3 = 2 (blue)
    tri_mask = bu.bfs(mask)

    
    if (np.sum(tri_mask == 0) < 10 or
            np.sum(tri_mask == 1) < 10 or
            np.sum(tri_mask == color) < 10):
        #print('Not filled')
        tri_mask = None

    if show:
        tri_img = np.stack([
            (tri_mask == 0)*255,
            (tri_mask == 1)*255,
            (tri_mask == 2)*255], axis=2).astype(np.uint8)
        print('Ground Truth Mask:')
        print('side0 = red, side1 = green, side2 = blue (border)')
        plt.imshow(tri_img); plt.show()
        
    if one_hot:
        vec_zero = (tri_mask == 0)
        vec_one = (tri_mask == 1)
        if color > 1:
            return np.stack([
                vec_zero,
                vec_one,
                tri_mask == color], axis=2).astype(np.uint8)
        else:
             return np.stack([
                vec_zero,
                vec_one]).astype(np.uint8)
    return tri_mask

def gmm_cluster(img, mask, gmm):
    '''
    make predictions, create mask with pred clusters
    '''

    pred1 = gmm.predict(img[mask == 1])
    pred0 = gmm.predict(img[mask == 0])
    pred2 = gmm.predict(img[mask == 2])
    img[mask == 1] = np.dstack([pred1]*3).squeeze() # poor mans rgb
    img[mask == 0] = np.dstack([pred0]*3).squeeze()
    img[mask == 2] = np.dstack([pred2]*3).squeeze()
    return img

def create_gmms(img, mask, values, n_components=3, random_state=12):
    '''
    For each value, a mixture model is created by indexing
    the image based on the values for which mask == value.
    args:
        img: (h, w, c) np.array image
        mask: (h, w) mask
        values: tuple of values in mask for which to create a mixture model
        n_components: number of components for each gmm. Should either be int
            or tuple with same length as values
    '''
    values = (values,) if type(values) == int else values
    n_components = (n_components,)*len(values) if type(n_components) == int else n_components
    assert len(values) == len(n_components)

    gmms = []
    for value, components in zip(values, n_components):
        gmms.append(
            GMM(
                n_components=components,
                random_state=random_state,
                value=value
            ).fit(img[mask == value])
        )
    return gmms

def gmm_test(img, mask, gmm, test_values=(1, 2), score=True, predict=False):
    '''
    Takes in a gmm trained on 1 part of a mask and runs inference on the other parts.
    args:
        img: original np.array image
        mask: (h,w) np.array with labeled portions
        order: a tuple where the elements are the values of different sections in the mask.
            first element of tuple is the values which the gmm was trained on.
    '''
    channel_apply = lambda img, mask, func: [func(img[mask == i]) for i in test_values]
    
    scores, res = None, None
    if score:
        scores = channel_apply(img, mask, gmm.score)
        

    if predict:
        pred0, pred1, pred2 = channel_apply(img, mask, gmm.predict)

        res = np.zeros_like(img)
        zeros = np.zeros_like(pred0)
        res[mask == gmm.value] = np.stack([pred0*255+255, zeros, zeros], axis=1)

        zeros = np.zeros_like(pred1)
        res[mask == test_values[0]] = np.stack([zeros, pred1*255, zeros], axis=1)

        zeros = np.zeros_like(pred2)
        res[mask == test_values[1]] = np.stack([pred2*255, pred2*255, zeros], axis=1)
    
    return scores, res
    
def gmm_gambit(gmms, img, mask, scores=True, predict=False):
    n = len(gmms)
    
    results = []
    perms = [(i, tuple(j for j in range(3) if i != j)) for i in range(3)]
    
    for perm in perms:
        train, tests = perm
        all_vals = [train, *tests]
        gmm = next(x for x in gmms if x.value == train)
        scores, preds = gmm_test(img, mask, gmm, test_values=all_vals, score=scores, predict=predict)
        results.append([all_vals, scores, preds])
            
    return results

def show_histograms(img, tri_mask):    
    color = ('r','g','b')
    sides = ['side0', 'side1', 'border']
    print('Color histograms for each piece in mask')
    for c in range(3):
        #img = border_img[tri_mask == c]

        print(sides[c], color[c])
        for i, col in enumerate(color):
            mask = (tri_mask==c).astype(np.uint8)*3
            histr = cv2.calcHist([img],[i],mask,[256],[0,256])
            #histr = cv2.calcHist([img],[0,1,2],mask,[100]*3,[0, 256]*3)
            #histr = np.bincount(img.ravel(),minlength=256)
            mcdonald = cv2.normalize(histr, histr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.show()
    plt.show()

def run_tri_gmm(jpeg_path, csv_path, save_samples=False, dims=(128, 128),
                n_components=3, border_thick=3):
    
    full_image = imageio.imread(jpeg_path)
    img = cv2.resize(full_image, dims)

    tri_mask = create_tri_mask(jpeg_path, old_dims=full_image.shape[:2], 
                               new_dims=dims, color=2,
                               thick=border_thick, show=False)
    if tri_mask is None:
        print('None for:', Path(jpeg_path).stem)
        return

    side0, side1, border = create_gmms(
        img,
        tri_mask,
        (0, 1, 2),
        n_components=n_components,
        random_state=12
    )

    # test all the things
    results = gmm_gambit(
        [side0, side1, border],
        img,
        tri_mask,
        scores=True,
        predict=False
    )

    # ********* Meta-stats below *************
    per_side_avg = np.zeros(3)
    per_side_max = np.zeros(3)
    
    

    csv_str = Path(jpeg_path).stem
    for n, res in enumerate(results):
        order, scores, preds = res
        csv_str += "," + ' '.join([str(x) for x in scores])
        base_val = scores[0]
        for i, o in enumerate(order): 
            if i == 0: continue
            diff = abs(abs(base_val) - abs(scores[i]))
            per_side_avg[n] += diff
            per_side_max[n] = max(diff, per_side_max[n])
        per_side_avg[n] /= len(scores[1:])
        #print('\n')

    total_avg = np.average(per_side_avg)
    total_max = np.max(per_side_max)

    if csv_path is not None:
        csv_str += f',{total_avg},{total_max}\n'

        if not os.path.exists(csv_path):
            csv_header = 'tile,side0,side1,border(side2),avg,max\n'
            os.makedirs(Path(csv_path).parent, exist_ok=True)
            with open(csv_path, 'w+') as f:
                f.write(csv_header)

        with open(csv_path, 'a') as f:
            f.write(csv_str)

if __name__ == "__main__":
    pass

