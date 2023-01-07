import sys, math
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path, PosixPath
from itertools import combinations
from collections import defaultdict
from pprint import pprint
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance as emd
from scipy.special import kl_div

PROJ_ROOT = Path(*Path.cwd().parts[:Path().cwd().parts.index('border-legibility')+1])

import models_mae
sys.path.append(str(PROJ_ROOT) + '/MODEL/GMM')
from gmmy_bear import create_tri_mask
sys.path.append(str(PROJ_ROOT) + '/data')
import border_utils as bu

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)() #<--- here is error
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

chkpt_dir = str(PROJ_ROOT) + '/MODEL/maskautoencode/mae/mae_visualize_vit_large_ganloss.pth'
model_mae_gan = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
cosine_distance = lambda x, y: 1 - torch.nn.CosineSimilarity(dim=0)(x, y)

def prep_img(img): #PIL image
    if type(img) == np.ndarray:
        img = cv2.resize(img, (224, 224))
    else:
        if type(img) in [str, PosixPath]:
            img = Image.open(img)
        img = img.resize((224, 224))
    img = np.array(img) / 255.

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    
    img = torch.tensor(img)
    
    # make it a batch-like
    img = img.unsqueeze(dim=0)
    img = torch.einsum('nhwc->nchw', img)
    return img.float()

def unprep_img(img):
    img = torch.einsum('nchw->hwc', img).detach().numpy()
    img *= imagenet_std
    img += imagenet_mean
    img *= 255
    return img.astype(np.uint8)

def reconstruct_plot(x, im_masked, y, im_paste):
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()

    
def make_mae_mask(img_file, side, thick=3):
    img = Image.open(img_file)
    h, w = img.size
    img = img.resize((224, 224))
    mask = create_tri_mask(img_file, (h,w), (224,224), color=2, thick=thick, show=False)
    
    mask2 = torch.tensor(mask != side).float()
    mask_3d = torch.stack([mask2] * 3, dim=2)
    mask_3d = torch.einsum('hwc->chw', mask_3d).unsqueeze(0)
    mask_3d = model_mae_gan.patchify(mask_3d)
    n, l, d = mask_3d.shape
    
    mask_1d = (mask_3d.sum(dim=2) < d).int()
    mask_2d = mask_1d.reshape(
        (n, int(math.sqrt(l)), int(math.sqrt(l)))
    )
    #bu.imshow(mask2, mask_2d.squeeze(), size=20)
    return mask_1d
    
def run_border_image(img_file, side, model, thick=3, mask_ratio=0, plot=True, rot=False):
    img = Image.open(img_file)
    x = prep_img(img)
    x_m = make_mae_mask(img_file, side, thick=thick)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    loss, y, mask = model(
        x.float().to(device), 
        x_m.float().to(device), 
        mask_ratio=mask_ratio
    )
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
        
    if plot:
        reconstruct_plot(x, im_masked, y, im_paste)
        
    if rot:
        # Rotate mask and output
        rot_mask = torch.rot90(mask, 1, [1, 2])
        rot_y = torch.rot90(y, 1, [1, 2])
        # create mask and pasted image with rotated mask and output
        rot_im_masked = x * (1 - rot_mask)
        rot_im_paste = x * (1 - rot_mask) + rot_y * rot_mask 
        if plot:
            reconstruct_plot(x, rot_im_masked, y, rot_im_paste)
   
        loss = torch.nn.MSELoss()(x, rot_im_paste)
        print('rot pasted:', loss)
     
    loss = torch.nn.MSELoss()(x, y)
    print('pasted:', torch.nn.MSELoss()(x, im_paste))
    return loss 

def run_one_image(img, model, ratio=0.75):
    x = prep_img(img)

    # run MAE
    loss, y, mask = model.forward_deprecated(x.float(), mask_ratio=ratio)
    y = model.unpatchify(y)
    res = y.clone()
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    #reconstruct_plot(x, im_masked, y, im_paste)
    return unprep_img(res) 
   
def run_expanding_border(img, border_mask, widths, side=0, plot=True):
    losses = torch.zeros(len(widths))
    for i, width in enumerate(widths):
        mask = bu.decode_segmap(border_mask, color=1, width=width, intensity=-1)
        side_img = img * (mask != side) + mask
        side_img = Image.fromarray(side_img)
        loss = run_border_image(img, side_img, model_mae_gan, mask_ratio=0, plot=plot)
        losses[i] = loss
    return losses

def run_increasing_mask(img, border_mask, rates, side=0, plot=True):
    losses = torch.zeros(len(rates))
    for i, rate in enumerate(rates):
        mask = bu.decode_segmap(border_mask, color=1, width=1, intensity=-1)
        side_img = img * (mask != side) + mask
        side_img = Image.fromarray(side_img)
        loss = run_border_image(img, side_img, model_mae_gan, mask_ratio=rate, plot=plot)
        losses[i] = loss
    return losses

def patch_compare_slow(x1, x2):
    res = torch.zeros(x1.shape[0], x2.shape[0])
    for p in range(x1.shape[0]):
        for p2 in range(x2.shape[0]):
            res[p][p2] = cosine_distance(x1[p], x2[p2])
    return res

def patch_compare(feats1, feats2):
    D, M = feats1.shape
    D, K = feats2.shape
        
    res = torch.zeros(M, K)
    # Vectorize all comparisons from one 
    for m in range(M):
        feat = feats1[:, m, None].repeat(1, K)
        dist = cosine_distance(feat, feats2)
        res[m] = dist 
    return res

def cosine_distance_gmm_img(img, vis=True):
    encoded, masks, regular = forward_encode_sides(img)
    
    distances = {i: [None, None, None] for i in range(3)}
    gmm_scores = {i: [None, None, None] for i in range(3)}
    gmms = [GaussianMixture(n_components=3, random_state=0)
                .fit(patch_compare(encoded[i].squeeze(), encoded[i].squeeze())
                        .mean(dim=1).detach().cpu().flatten().reshape(-1, 1)
                    )
               for i in range(3)
   ]
    
    # compare each i to every j
    combos = list((i, j) for i in range(3) for j in range(3) if i != j)
    #combos = combinations(range(3), r=2)
    
    for i, j in combos:
        x1, x2 = [torch.einsum('npd->pd', x) for x in [encoded[i], encoded[j]]]
        # Sum along last channel
        #x1, x2 = [torch.einsum('pd->p', torch.tensor(x)).unsqueeze(0) for x in [x1, x2]]
        
        
        
        def save_compare(a, b, i_a, i_b):
            if distances[i_a][i_b] is not None: return
            res_all = patch_compare_slow(a, b)        
            res_some = res_all.mean(dim=1) # compare distance of patch in A to all in B
            res = res_some.mean() # combine distances for each patch in A
            distances[i_a][i_b] = res
            
            if gmm_scores[i_a][i_b] is not None: return
            
            score = gmms[i_a].score(
                res_some.detach().cpu().numpy().flatten().reshape(-1, 1)
            )
            gmm_scores[i_a][i_b] = score
            
            
        save_compare(x1, x2, i, j)
        save_compare(x2, x1, j, i)
    
    scores = {
        k: {
            i: round(v[i], 6) 
                for i in range(len(v)) if v[i] is not None 
        } for k, v in gmm_scores.items()
    }
    
    distances = {
        k: {
            i: round(v[i].item(), 6)
                for i in range(len(v)) if v[i] is not None
        } for k, v in distances.items()
    }
    if vis:
        print('image 0', '\t' * 3, 'image 1', '\t' * 3, 'image 2')
        print('Distance from image at index i to index j: Smaller is closer')
        pprint(distances)
        print('''GMM log likelihood of distance distribution from image i -> image j
        given distance distribution image i -> image i: larger is more similar''')
        pprint(scores)
    max_diff = 0
    for score_dict in scores.values():
        score_list = list(score_dict.values())
        try:
            max_diff = max(max_diff, abs(min(score_list) - max(score_list)))
        except TypeError:
            print("None value detected")
    return max_diff

def cosine_distance_img(img, vis=True):
    encoded, masks, regular = forward_encode_sides(img)
    
    distances = {}
    
    # compare each i to every j
    #combos = list((i, j) for i in range(3) for j in range(3))
    combos = combinations(range(3), r=2)
    
    for i, j in combos:
        x1, x2 = [torch.einsum('npd->pd', x) for x in [encoded[i], encoded[j]]]
        
        res_all = patch_compare(x1, x2)        
        res = res_all.mean(dim=1) # compare distance of patch in A to all in B
        res = res.mean() # combine distances for each patch in A
        distances[f'{i}->{j}'] = res

    return max([abs(a-b) for a in distances.values() for b in distances.values()])


def forward_encode_sides(img_file, model=model_mae_gan):
    img = Image.open(img_file)#.crop(box=(0, 0, 1200, 1200))
    h, w = img.size 
    
    x = prep_img(img)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    encoded = []
    latent_masks = []
    for side in range(3):
        x_m = make_mae_mask(img_file, side, thick=3)
        x_encoded, mask, ids_restore = model.forward_encoder(
            x.float().to(device), 
            x_m.float().to(device), 
            mask_ratio=0,
            norm=False
        )
        encoded.append(x_encoded)
        latent_masks.append(mask)
    return encoded, latent_masks, [0, 1, 2]

def forward_encode_whole(img_file, dims=(1280, 1280), thick=10):
    img = Image.open(img_file)#.crop(box=(0, 0, 1200, 1200))
    h, w = img.size
    border_mask = create_tri_mask(img_file, (h,w), (h,w), color=2, thick=thick, show=False)#[:1200, :1200]
    
    side_0 = img * np.stack([border_mask == 0] * 3, axis=2)
    side_1 = img * np.stack([border_mask == 1] * 3, axis=2)
    side_2 = img * np.stack([border_mask == 2] * 3, axis=2)
    img = img * np.stack([border_mask > -1] * 3, axis=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_mae_gan.to(device)
    x = prep_img(img)
    patched = model_mae_gan.patchify(x)
    encoded, mask, ids = model_mae_gan.forward_encoder_deprecated(
        x.float().to(device), 0)
    
    masks = []
    all_indices = []
    #encoded_sides = []
    for side in [side_0, side_1, side_2]:
        side = prep_img(side)
        p = model_mae_gan.patchify(side)
        similarity = (p != patched).sum(dim=2) # NxL counting differences in D
        indices = (similarity == 0).long() # NxL where 1 is patches were the same else 0
        masks.append(similarity)
        all_indices.append(indices)
        #encoded_sides.append(encoded[0,:-1,:][indices[0, :] > 0])
    return encoded, masks, all_indices

def distribution_kmeans_preds(img, comp_func, n_clusters=3, vis=False):
    img_file = img
    if type(img) in [str, PosixPath]:
        img = bu.imread(img)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=600, random_state=12)

    encoded, masks, ids = forward_encode_whole(img_file, thick=3)
    ids[2] = masks[2] < masks[2].max()
    
    # -1 for class token, use [0] index instead of torch.squeeze
#     encode = encoded[0,:-1,:].detach().numpy()
    
    '''dont use'''
    encode = torch.Tensor.cpu(encoded[0,:-1,:]).detach().numpy()
    
    n, d = encode.shape
    #encode = pca.fit_transform(encode)
    fitted = kmeans.fit(encode)

    all_preds = fitted.labels_
    cluster_img = all_preds.reshape((int(np.sqrt(n)),int(np.sqrt(n))))
    
    if vis:
        mask = np.stack([ids[0]*0, ids[1], ids[2]*2], axis=2) \
               .squeeze() \
               .max(axis=1) \
               .reshape((int(np.sqrt(n)),int(np.sqrt(n))))
        bu.imshow(regular, cluster_img, mask, size=16)
    scores = {}
    centers = {}

    eps = 1e-8

    for idx, c in zip(ids,range(3)):
        # -1 for class token, use [0] index instead of torch.squeeze
        patches = encode[idx[0, :] > 0]
#         patches = pca.transform(patches)
        try:
            preds = fitted.predict(patches)
        except ValueError:
            return 0
        n = len(preds)
        scores[c] = np.array([
            np.clip((preds == i).sum() / n, eps, 1-eps) for i in range(n_clusters)
        ])
        
        # compare cluster distances if no func
        if comp_func is None:
            centers[c] = KMeans(n_clusters=1, n_init=50, max_iter=600, random_state=12) \
                        .fit(patches) \
                        .cluster_centers_ \
                        .mean()

    res = None
    if comp_func is not None:
        res = {f'{k1}->{k2}': comp_func(v1, v2) for (k1, v1), (k2, v2) in combinations(scores.items(), r=2)}
    else:
        res = centers
        
    if vis:
        if comp_func is not None:
            pprint(res)
        else:
            print('centers')
            pprint(centers)
            print('scores')
        pprint(scores)
        print('\n')
    return max([abs(a-b) for a in res.values() for b in res.values()])
