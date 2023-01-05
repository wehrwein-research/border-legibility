import warnings
import sys, os, json, csv, math, random
from pathlib import Path, PosixPath
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
from collections import deque
from shapely.geometry import *
from shapely import affinity
from sultan.api import Sultan as Bash
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from skimage.morphology import flood_fill



def bfs_slow(mask, fill_value=1):
    assert fill_value >= 1
    x, y = np.nonzero(mask == 0)
    index = (x[0], y[0])
    H,W = mask.shape
    queue = deque([index])
    while len(queue) > 0:        
        h,w = queue.pop()
        if h >= 0 and w >= 0 and h < H and w < W and mask[h][w] < 1:            
            mask[h][w] = fill_value
            queue.appendleft((h + 1, w))
            queue.appendleft((h - 1, w))
            queue.appendleft((h, w + 1))
            queue.appendleft((h, w - 1))
            
    return mask  

def bfs(mask, fill_value=1):
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    assert fill_value >= 1
    h, w = mask.shape
    r = lambda x: random.randrange(0, x)
    i, j = r(h), r(w)
    # start at top left and move down until we find a zero value
    while mask[i][j] != 0:
        i, j = r(h), r(w)
    print("i: " + str(i))
    print("j: " + str(j))
        
    return flood_fill(mask, (i, j), fill_value, connectivity=1)

def imread(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise Exception('Image does not exist')
    
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def is_bad_bing_image(image):
    if type(image) in [PosixPath, str]:
        image = imread(image)
    
    # The unique color used by bad images
    death_pixel = np.array([245, 242, 238], dtype=np.uint8)

    # Number of pixels with same weird beige color as the bad images
    death_count = np.sum(np.all(image == death_pixel, axis=2))

    total_count = image.shape[0] * image.shape[1]
    # Don't use image if it is over 50% weird beige color
    return (death_count / total_count) > 0.5

def check_bounds(p1, p2, img_dims):
    X, Y = img_dims
    p1_invalid = p1[0] < 0 or p1[0] > X or p1[1] < 0 or p1[1] > Y
    p2_invalid = p2[0] < 0 or p2[0] > X or p2[1] < 0 or p2[1] > Y
    
    both = all((p1_invalid, p2_invalid))
    one = any((p1_invalid, p2_invalid))
    if one:
        a, b = get_border_endpoints(cv2.line(np.zeros(img_dims), p1, p2, 1))
        if a is None and b is None:
            raise Exception("None found {}, {}".format(p1, p2))
        if both:
            p1 = (a.x, a.y)
            p2 = (b.x, b.y)
        else:
            p = a if a is not None else b
            if p1_invalid:
                p1 = (p.x, p.y)
            else:
                p2 = (p.x, p.y)
    return p1, p2
            
    
def is_point_on_segment(point, segment):
    '''
    Finds out if a given point exists on the given line segment
    point: [x, y]
    segment: [x,y, x,y]
    '''
    # help account for long gps coordinate float point errors
    buffer = 5e-5
    p1, p2 = segment[0:2], segment[2:]
    
    min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
    min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
    
    # Check that the bounds of the point are within the bounds of the segment
    valid_x = min_x-buffer <= point[0] <= max_x+buffer
    valid_y = min_y-buffer <= point[1] <= max_y+buffer
    if not valid_x or not valid_y:
        return False
    
    y, x = max_y - min_y, max_x - min_x
    
    # account for divide by zero. If x is zero and y is valid, the point is on a vertical segment
    if x == 0: return valid_y
    
    slope = y / x
    intercept = p1[1] - slope * p1[0]
    # solve y = mx + b. If the point is on the line, and has valid bounds, it is on the segment
    return abs(point[1] - (slope * point[0] + intercept)) < buffer

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def get_bbox_coord_intersect(bbox, coords):
    '''
    
    args:
        bbox: [minlon, minlat, maxlon, maxlat]
        coords: [(lat1, lon1), (lat2, lon2)]
        
    return:
        LineString containing the coordinates of the 
        intersection with the bbox. If there is no
        intersection, we return None
    '''
    b, l, t, r = bbox  
    shapely_box = box(l, b, r, t)
    shapely_line = LineString([coords[0][::-1], coords[1][::-1]])
    inter = shapely_line.intersection(shapely_box)
    return inter if inter.length > 0 else None

def dist(bbox, coords):
    b, l, t, r = bbox
    bbox = [(l,b), (r,b), (l,t),(r,t)]
    p = LineString(coords)
    b = box(l, b, r, t, ccw=True)
    return p.distance(b)

    
def are_coords_in_box_range(coords, bbox):
    """
    Takes in 2 coordinates and a coordinate bounding box to 
    find out if any point on the line between the two 'coords'
    can be located in 'bbox'. Essentially checking if the line
    created by the coordinates intersects the bounding box coordinates
    """
    A, B = coords
    b, t = min(A[0], B[0]), max(A[0], B[0])
    l, r = min(A[1], B[1]), max(A[1], B[1])
    bottom, left, top, right = bbox
    vertical = [bottom, top]
    horizontal = [left, right]
    
    # Create a list of the four corners of bounding box
    corners = [[x, y] for y in vertical for x in horizontal]

    # Create 4 line segments from the corners
    segments = [[p1, p2] for p1, p2 in itertools.combinations(corners, 2) 
                if p1[0] == p2[0] or p1[1] == p2[1]]
    
    # Find intersections between each segment and the given coordinates
    for p1, p2 in segments:
        # change lat, lon to x, y
        p1, p2 = p1[::-1], p2[::-1]
        p3 = get_intersect(A, B, p1, p2)
        
        # Line segment is valid if p3 is on the segment or if the segment is box
        valid_coord_seg = is_point_on_segment(p3, [*A,*B]) or \
            ((bottom <= b <= top or bottom <= t <= top) and (left <= l <= right or left <= r <= right))
        
        # Check if p3 occurs on any of the 4 box segments
        valid_box_seg = any([is_point_on_segment(p3, [*x[::-1], *y[::-1]]) for x,y in segments])
        if valid_coord_seg and valid_box_seg:
            return True
    return False

    
    
def get_border_endpoints(mask):
    r = len(mask) - 1
    c = len(mask[0]) - 1
    
    a, b = None, None
    top = np.where(mask[0]==1)
    if len(top[0]):
        b = Point(top[0][0], 0) if a is not None else b
        a = Point(top[0][0], 0) if a is None else a
    
    bottom = np.where(mask[-1]==1)
    if len(bottom[0]):
        b = Point(bottom[0][0], r) if a is not None else b
        a = Point(bottom[0][0], r) if a is None else a
        
    left = np.where(mask[:,0]==1)
    if len(left[0]):
        b = Point(0, left[0][0]) if a is not None else b
        a = Point(0, left[0][0]) if a is None else a
        
    right = np.where(mask[:,-1]==1)
    if len(right[0]):
        b = Point(c, right[0][0]) if a is not None else b
        a = Point(c, right[0][0]) if a is None else a
    
    left, right = None, None
       
    if a is None or b is None:
        left = a
        right = b
    else:
        left = a if a.x < b.x else b
        right = b if b.x > a.x else a
    
    return left, right
        
def ax_by_calc(mask):
    """ Returns Ax + By + C = 0 paramatization"""
    left, right = get_border_endpoints(mask)
    if not left or not right:
        return None, None
    
    A = (left.y - right.y) / len(mask)
    B = (left.x - right.x) / len(mask[0]) 
    C = -1 * (A * (left.x / len(mask[0])) + B * (left.y / len(mask)))
    return A, B, C

def beta_angle_calc(p1, p2):
    buffer = .00001
    opp = p2.x - p1.x
    adj = (p1.y - p2.y) + buffer
    beta = math.degrees(math.atan(opp / adj))
    isNeg = -1 if beta < 0 else 1
    return (90 - abs(beta)) * isNeg 

def beta_angle(mask):
    p1, p2 = get_border_endpoints(mask)
    if not p1 or not p2:
        return None, None
    else:
        return beta_angle_calc(p1, p2)
    
def mask_from_segments(lines, dims, draw_lines=True, color=1, thick=1):
    mask = np.zeros((dims))
    for line in lines:
        x1, y1, x2, y2 = line
        if draw_lines:
            mask = cv2.line(mask, (x1, y1), (x2, y2), color, thick)
        else:
            mask = cv2.line(mask, (x1, y1), (x1, y1), color, thick)
            mask = cv2.line(mask, (x2, y2), (x2, y2), color, thick)
    return mask
    
    
def decode_segmap(image, color=1, width=5, intensity=-1):
    """
    Turns a segmentation map into a rgb pic with green visualizaitons 
    """
    colored = ((image > 0) * 255).astype(np.uint8)
    w = np.where(colored == 255)
    y = list(w[0])
    x = list(w[1])
    for a,b in zip(x,y):
        colored = cv2.circle(colored, (a,b), width, 255, intensity)
    z = np.zeros_like(image).astype(np.uint8)
    stack = [z,z,z]
    stack[color] = colored
    rgb = np.stack(stack, axis=2)
    return rgb


def imshow(*imgs, size=4, per_line=6):
    """
    Plots images side by side
    """
    n = len(imgs)
    
    lines = int(np.ceil(n / per_line))
    imgs = list(imgs)
    for line in range(lines):
        plt.rcParams['figure.figsize'] = [size, size]
        f = plt.figure()
        
        for i, img in enumerate(imgs[per_line*line:per_line*(line+1)]):
            f.add_subplot(1, per_line, i+1)
            plt.imshow(img)        
        plt.show(block=False);plt.clf()
    
def resize_segments(segs, old_dims, new_dims):
    delta_x =  new_dims[1] / old_dims[1]
    delta_y = new_dims[0] / old_dims[0]
    for i, seg in enumerate(segs):
        x1, y1, x2, y2 = seg
        segs[i][0] = int(x1 * delta_x)
        segs[i][1] = int(y1 * delta_x)
        segs[i][2] = int(x2 * delta_y)
        segs[i][3] = int(y2 * delta_y)

    return segs

def plot_segs(segs, show=True):
    for s in segs:
        x, y = s.xy
        plt.plot(x, y, 'r')
    if show:
        plt.gca().invert_yaxis()
        plt.show()

def segments_to_shapely(segs):
    reshaped = [((s[0], s[1]), (s[2], s[3])) for s in segs]
    return MultiLineString(reshaped)

def shapely_to_segments(segs):
    res = []
    for seg in segs:
        res.append(np.array(seg.coords).flatten().astype(np.uint16))
    return np.array(res, dtype=np.int64)

def rotate_segments(segs, theta, bounds, clip=True, vis=False):
    assert len(bounds) == 4
    segs = segments_to_shapely(segs)
    
    if vis:
        for s in segs:
            x, y = s.xy
            plt.plot(x, y)
        plt.gca().invert_yaxis()
        if bounds: 
            plt.xlim([bounds[1], bounds[3]])
            plt.ylim([bounds[0], bounds[2]])
        plt.show()
        
    center = (bounds[3]-bounds[1])/2, (bounds[2]-bounds[0])/2
    segs = affinity.rotate(segs, theta, origin=center)
    
    if clip:
        shape_box = box(*bounds)
        # Clip segments into the range of the bounding box
        segs = [s.intersection(shape_box) for s in segs if shape_box.contains(s)]

    if vis:
        for s in segs:
            x, y = s.xy
            plt.plot(x, y)
            
        plt.gca().invert_yaxis()
        if bounds: 
            plt.xlim([bounds[1], bounds[3]])
            plt.ylim([bounds[0], bounds[2]])
        plt.show()
        
    segs = shapely_to_segments(segs)
    return segs
    
def getCode(name, file):
    """
    Pass in the country name, get back the code.

    Input:
        name: the correctly spelled an capitalized country  name
        file: the file object of the COW csv file
    Output:
        name: a tuple of a 3 letter abbreviation string and full name string for a country
    """
    if name.islower():
        name = name.upper()
    size = 243
    code = 0
    #244 lines of fun to traverse:
    for i in range(size):
        if file['StateNme'][i] == name or file['StateAbb'][i] == name:
            code = file['CCode'][i]
            break
            
    return code

def get_shape_line(land1, land2, borderFile, cow):
    """
    Takes a couple of strings of country names and gives you
    back what line of the shapefile to start reading on

    Input:
        land1: Correctly spelled and capitalized country name
        land2: A different correctly spelled country name that has a
               land border in common with land1
        borderFile: the shapefile where the borders live
        cow: the Correlates of War csv file with all of the country codes
    Output:
        line: an int that represents the shapefile land border between
              land1 and land2
    """

    line = -1

    size = 319

    l1c = getCode(land1, cow)
    l2c = getCode(land2, cow)
    for i in range(size):
        if borderFile['LEFT_FID'][i] == l1c:
            if borderFile['RIGHT_FID'][i] == l2c:
                line = i
        if borderFile['LEFT_FID'][i] == l2c:
            if borderFile['RIGHT_FID'][i] == l1c:
                line = i
    if line < 0:
        raise Exception("Are you sure these two have a land border?")

    return line

def get_random_tile(root='../bing_maps/', 
                    folder='global-scrape', no_bing_err=True, 
                    multi_color=False, has_segments=False, 
                    tri_mask=False):
    base_dir = Path(root, folder)
    country_border_folders = [x for x in base_dir.iterdir() \
                              if (base_dir / x).is_dir()]
    img_path = None
    valid = False
    num_tries = 0
    while not valid and num_tries < 20:
        border_folder = random.choice(country_border_folders)
        all_tile_folders = os.listdir(Path(base_dir, border_folder))
        tile_folder = random.choice(all_tile_folders)
        
        img_path = Path(base_dir, border_folder, tile_folder, tile_folder).with_suffix('.jpeg')
        
        err = verify_tile_integrity(img_path, skip_tri_mask=not tri_mask)
        
        valid = not (
            (err == 1) or                   # path doesn't exist
            (err == 2 and no_bing_err) or   # bing error when requested
            (err == 3 and has_segments) or  # no segment file when requested
            (err == 4 and multi_color) or   # img all one color when requested
            (5 <= err <= 6 and tri_mask)    # no tri mask when requested
        )
        num_tries += 1
        
    assert num_tries < 20 # Couldn't find a valid image 20 times in a row 
    return str(img_path)

''' ERROR CODES: 
    0: No error
    1: Img doesn't exist
    2: Default Bing error image
    3: Img is all one color (with variance arg 'pixel_diff')
    4: Corresponding line segment file doesn't exist
    5: Tri-mask can't be made (likely no lines in numpy file)
    6: Tri-mask not fillable (min pixels to be considered filled is arg 'min_tri_mask_pix')
'''
def verify_tile_integrity(img_path, ret_as_list=False, skip_tri_mask=False, 
                          pixel_diff=5, min_tri_mask_pix=50, border_thic=10):
    img_path = Path(img_path) 
    errors = []
    
    if not img_path.exists():
        if not ret_as_list:
            return 1
        errors.append(1)
        
    if is_bad_bing_image(img_path):
        if not ret_as_list:
            return 2
        errors.append(2)
    
    img = imread(img_path)
    if np.allclose(img, img.mean(), rtol=0, atol=pixel_diff):
        if not ret_as_list:
            return 3
        errors.append(3)
    
    if not img_path.with_suffix('.npy').exists():
        if not ret_as_list:
            return 4
        errors.append(4)
    
    if skip_tri_mask:
        if not ret_as_list:
            return 0
        return errors 
    ''' try to create tri-mask '''
    lines = np.load(img_path.with_suffix('.npy'))
    
    mask = None 
    try:
        mask = mask_from_segments(lines, dims=img.shape[:-1], draw_lines=True,
                                 color=2, thick=border_thic)
    except Exception as e:
        print(e)

    # side1 = 0 (red), side2 = 1 (green), side3 = 2 (blue)
    tri_mask = None
    if mask is not None:
        try:
            tri_mask = bfs(mask)
        except Exception as e:
            print(e)
    if tri_mask is None: 
        if not ret_as_list:
            return 5
        errors.append(5)
        return errors
        
    if (np.sum(tri_mask == 0) < min_tri_mask_pix or
            np.sum(tri_mask == 1) < min_tri_mask_pix or
            np.sum(tri_mask == 2) < min_tri_mask_pix):
        if not ret_as_list:
            return 6
        errors.append(6)

    if not ret_as_list: 
        return 0
    return errors 

# Split tile into sub crops of size (crop_size)
# grouped by side0, side1, and border 
def get_crops_from_tile(tile, tile_size=(1280, 1280), crop_size=(256, 256)):
    tile = Path(tile)
    lines = np.load(tile.with_suffix('.npy'))

    h, w = tile_size
    mask = mask_from_segments(lines, tile_size, draw_lines=True, color=2, thick=1)

    c_h, c_w = crop_size
    
    # Create a grid of crops that fills the dimensions of the tile
    n_crops_w = w // c_w
    n_crops_h = h // c_h
    boxes = [
        [0 + c_w * d_w, 0 + c_h * d_h, c_w + c_w * d_w, c_h + c_h * d_h] 
        for d_w in range(n_crops_w)
        for d_h in range(n_crops_h)
    ]

    filled_mask = bfs(mask)
    
    border_boxes = []
    side0_boxes = []
    side1_boxes = []
    for boxx in boxes:
        l, b, r, t = boxx
        boxx = b, l, t, r
        area = filled_mask[b:t,l:r]
        has_border = np.any(area == 2)

        if has_border:
            border_boxes.append(boxx)
        elif np.any(area == 1):
            side1_boxes.append(boxx)
        else:
            side0_boxes.append(boxx)
    
    return side0_boxes, side1_boxes, border_boxes

def vis_vectorfield(afm, sub=None, lines=None, proj=False):
    indices = np.array(np.mgrid[:afm.shape[1], :afm.shape[2]])
    if not proj:
        afm = afm + indices
        
    if afm.shape[-1] != 2:
        h, w = afm.shape[1:]
        afm = afm.transpose((2,1,0)).reshape(h*w, 2)

    afm = afm[::sub,:]
    plt.plot([x[0] for x in afm], [x[1] for x in afm],'bo')
    plt.xlim([0, np.sqrt(afm.shape[0])])
    plt.ylim([0, np.sqrt(afm.shape[0])])
    if lines is not None:
        for line in lines:
            plt.plot([line[0],line[2]],[line[1],line[3]],'y')
    plt.gca().invert_yaxis()
    plt.show()

def show_box_plot(bbox=None, buf=0.008, plot=True):
    if bbox is not None:
        b, l, t, r = bbox
        b, l, t, r = [b-buf, l-buf, t+buf, r+buf]
        plt.ylim(b, t)
        plt.xlim(l, r)
    if plot:
        plt.show()
        plt.clf()

def grep_file(file, terms):
    '''
        file: (str) path to file you are trying to find text in
        terms: (list of str) terms you want to find inside of file
        
        return: (list of str) lines from file that contain one of the terms
    '''
    # grep_str example: 'usa-can' | 'usa-mex' | other_countries... 
    grep_str = "\'" + r"\|".join(terms) + "\'"
    bash = Bash()
    all_files = bash.cat(file).pipe().grep(grep_str).run().stdout
    return all_files

def shuf_file(file, n):
    bash = Bash()
    files = bash.shuf(f'{file} -n {n}').run().stdout
    return files

if __name__ == '__main__':
    pass