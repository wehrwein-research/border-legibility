import shapefile
from pykml import parser
from pykml.util import count_elements
import sys, os, json, csv, math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import cv2
from border_utils import *

sys.path.append('../bing_maps/scrape')
from bing_coords import getBBox

SHP_FILE = '../bing_maps/scrape/Borders2/Int_Borders_version_2.shp'
DOC_KEY = 'http://www.opengis.net/kml/2.2'
COW = '../bing_maps/scrape/cow-country-code.csv'

def bbox_csv_from_txt(txt_path, csv_path, root):
    with open(txt_path, 'r') as t:
        with open(csv_path, 'w+') as c:
            c.write('filename,bbox\n')
            for txt_line in t:
                txt_line = txt_line.strip()
                b, l, t, r = getBBox(
                    (root / txt_line).with_suffix('.json'),
                    False
                )
                bbox_str = f'{b} {l} {t} {r}'
                c.write(f'{txt_line},{bbox_str}\n')

    
def coord_to_pixel(coord, bbox, dims):
    """
    Uses the ratio between a coordinate and its bounding box
    to a respective pixel mapping. Requires known dimensions
    for the image
    """
    bottom, left, top, right = bbox
    lat, lon = coord
    H, W = dims
    
    gps_height = abs(top - bottom)
    gps_width = abs(right - left)
    lat = lat - bottom
    lon = lon - left
    
    h = lat / gps_height
    w = lon / gps_width
    
    pixel_w = int(w * W)
    pixel_h = H - int(h * H)
    
    return pixel_w, pixel_h


def add_coords_to_mask(coords, mask, bbox, verbose):
    """
    Given two coordinates, a gps bounding box, and a 2d numpy array,
    we convert the coordinates to pixels, then mark each pixel that
    is between the two points
    """
    start = coord_to_pixel(coords[0], bbox, mask.shape)
    end = coord_to_pixel(coords[1], bbox, mask.shape)
    if verbose:
        print('og coords:', coords)
        print('pixels:', start, end)
    mask = cv2.line(mask, start, end, 1)
    return mask



def iter_shp_coords(shp_file, country1=None, country2=None):
    """
    Returns an iterator that outputs 2 pairs of coordinates at a time
    shp_file: a file path to a .shp file
    country1: a string name of the first of the border countries. Either properly capitalized full name or 3 letter abbreviation
    country2: a string name of the second border country. Either properly capitalized full name or 3 letter abbreviation
    
    Output Format: ( (lat1, lon1), (lat2, lon2) ) -> ( (lat2, lon2), (lat3, lon3)) -> ...
    """ 
    shapes = None
    
    sf = shapefile.Reader(shp_file)
    if country1 is not None: 
        f = pd.read_csv(COW)
        borderFile = gpd.read_file(shp_file)
        i = get_shape_line(country1, country2, borderFile, f)
        shapes = [sf.shapes()[i]]
    else:
        shapes = sf.iterShapes()
    
    for shape in shapes:
        temp = None
        for point in shape.points:
            if temp is not None:
                lon1, lat1 = temp
                lon2, lat2 = point
                yield [lat1, lon1], [lat2, lon2]
            temp = point 

def iter_kml_coords(kml_file, country1=None, country2=None):
    """
    Returns an iterator that outputs 2 pairs of coordinates at a time ( (lat1, lon1), (lat2, lon2) )
    
    """
    with open(kml_file) as f:
        kml = parser.parse(f)
        root = kml.getroot()
        folder = root.Document.Folder.Placemark
        # placemarks are usually states/provinces
        for placemark in folder:
            name_root = placemark.ExtendedData.SchemaData.SimpleData
            name = None
            # each placemark has a name for its country and the specific landmark
            for child in name_root:
                if child.get('name') == 'NAME_1':
                    name = child
            if not name:
                continue

            if BORDER_STATES and name not in BORDER_STATES:
                continue
            polygons = placemark.MultiGeometry.Polygon
            for poly in polygons:
                coords = poly.outerBoundaryIs.LinearRing.coordinates
                # coords is a list of strings
                for coord in coords:
                    # split string into more strings(now pairs of csv coordinates) 
                    coord = coord.text.split(',0 ')
                    temp = None
                    for c in coord:
                        c = c.split(',')
                        # ensuring coordinates have 2 values
                        if len(c) == 2:
                            # serve the coordinates in pairs of two, using temp to store previous values
                            if temp is not None:
                                lon, lat = c
                                t_lon, t_lat = temp
                                if 48.5 < float(t_lat) < 49.5 and -102.5 < float(t_lon) < -102:
                                    print(temp)
                                yield (float(t_lat), float(t_lon)), (float(lat), float(lon))
                            temp = c


def make_mask(img, bbox, data_file, country1, country2, verbose=0):
    """
    Returns a H*W mask of border pixels if the image has border coordinates
    located within kml file, else returns None. 
    """
    mask = None
    iter_coords = iter_kml_coords if data_file.endswith('kml') else iter_shp_coords
    for coords in iter_coords(data_file, country1, country2):
        if verbose == 2:
            print('Candidate:', coords)
        are_valid_coords = are_coords_in_box_range(coords, bbox)
        if are_valid_coords:
            if verbose == 2:
                print('Match:', coords, '\n************************************')
            if mask is None:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = add_coords_to_mask(coords, mask, bbox, verbose)
    return mask

def get_lines_from_json(json_path, shp_file, dims=(256,256), vis=False):
    '''
    Creates line segments for the pieces of the border that pass through a given tile.

    args:
        json_path: The string file path to a json file for a unique file
        shp_file: The string file path to the ground truth .shp file with the border coordinates
        dims: Image dimensions for the tile
        vis: If True, we plot the line segments in matplotlib
    
    return:
        segments: Numpy array of line segments of shape (N, 4)
    '''
    
    if is_bad_bing_image(json_path.with_suffix('.jpeg')):
        return np.array([])
    # get country abbreviations from string path
    c1, c2 = Path(json_path).stem[:7].strip().split('-')
    bbox = None
    try:
        # get bounding box for quadcode tiles
        bbox = getBBox(json_path, coordFormat=False)
    except Exception as e:
        # get bounding box from json for 1280x1280 
        bbox = json.load(open(json_path))['resourceSets'][0]['resources'][0]['bbox']
   
    # coord shape: [(lat1, lon1), (lat2, lon2)]
    coords = []
    # Iterate over every coord_pair in the border between c1 and c2 
    for coord in iter_shp_coords(shp_file, country1=c1, country2=c2):
        # Find intersection between coordinates and bounding box
        intersect = get_bbox_coord_intersect(bbox, coord) # returns shapely object

        # plot coordinates
        if vis:
            shapely_line = LineString([coord[0][::-1], coord[1][::-1]])
            plt.plot(*shapely_line.coords.xy, 'b')
            
        # turn shapely object into pixel coordinates 
        if intersect is not None:
            # shape ([x1, x2], [y1, y2])
            xs, ys = intersect.coords.xy
            # -> shape [(x1, y1), (x2, y2)]
            intersect_coord = [(x,y) for x, y in zip(xs, ys)]
            # gps coordinates -> pixel coordinates
            pixels = [coord_to_pixel(c[::-1], bbox, dims) 
                                  for c in intersect_coord]
            # -> shape [x1, y1, x2, y2]
            pixels = [*pixels[0], *pixels[1]]
            coords.append(pixels)
    
    coords = np.array(coords)
    
    # Plot bounding box and full image with border drawn on it
    if vis:        
        # print coords
        b, l, t, r = bbox  
        print(Path(json_path).stem,':', (b,l), (t,l), (t,r), (b,r))
       
        # plot bounding box
        b = box(l,b,r,t)
        plt.plot(*b.exterior.coords.xy, 'r')
        show_box_plot(bbox)
        
        # plot image with lines drawn
        img = imageio.imread(Path(json_path).with_suffix('.jpeg'))
        mask = mask_from_segments(coords, dims, draw_lines=True)
        rgb = decode_segmap(mask, width=1)
        annotated = img * (rgb < 255) + rgb
        plt.imshow(annotated);plt.show();plt.clf()
    return coords 

def get_bbox_from_tile(json_file, verbose=False):
    if os.path.exists(json_file):
        with open(json_file) as f:
            meta = json.load(f)
            bb = meta["resourceSets"][0]["resources"][0]["bbox"]
            if verbose:
                print('tile bot-left, top-right:', bb[:2], bb[2:4])
    else:
        return None
    return bb

def get_bbox_universal(json_path, center=False):
    bbox, center = None, None
    try:
        json_values = json.load(open(json_path))
        bbox = json_values['resourceSets'][0]['resources'][0]['bbox']
        center = json_values['resourceSets'][0]['resources'][0]['mapCenter']['coordinates']
    except:
        b, l, t, r = getBBox(json_path, False)
        center = (t + b) / 2, (r + l) / 2
    
    return bbox, center if center else bbox


def segment_folder(dataset_path, folder, dims, 
                   shp_file=SHP_FILE, vis=False, save=True):
    '''
    Creates and saves line segments to a .npy file for each tile
    in a dataset.
    '''
    border_path = Path(dataset_path, folder)
    if len(folder) != 7 or not folder[0].isalpha(): 
        print(folder, 'is not a border folder.')
        return

    for name in os.listdir(border_path):
        json_file = Path(border_path, name, name + '.json')
        np_file = Path(border_path, name, name + '.npy')
        if np_file.exists() or not json_file.exists():
            continue

        try:
            lines = get_lines_from_json(
                        json_file,
                        dims=dims,
                        shp_file=shp_file,
                        vis=vis
                    )
            if len(lines) > 0 and save:
                np.save(np_file, lines)
            else:
                print('Nothing for:', name)
        except Exception as e:
            print(e)

def mask_dir(IMG_PATH, data_file, verbose=False):
    """
    Creates a mask for each border picture in a given directory path
    that has gps meta data. Masks are H*W with borders marked with 
    the number 1.
    """
        
    for tile in os.listdir(IMG_PATH):
        json_file = Path(IMG_PATH, tile, tile + ".json")
        bb = get_bbox_from_tile(json, verbose)
        if bb is not None:
            img = imageio.imread(Path(IMG_PATH, tile, tile + '.jpeg'))
            mask = None
            try:
                mask = imageio.imread(Path(IMG_PATH, tile, 'mask.png'))
                print('mask found at:', IMG_PATH, tile)
                continue
            except:
                mask = make_mask(img, bb, data_file, verbose)

            if mask is not None:
                #slope, intercept = lin_approx(mask)
                print(tile, mask.shape, np.count_nonzero(mask))
                #print(f'Line approximation: y = ({slope})x + {intercept}')
                #imshow(img, decode_segmap(mask))
                imageio.imwrite(Path(IMG_PATH, tile, 'mask.png'), mask)
            else:
                print('No mask at:', tile, '\n', bb)
        else:
            print('No json at:', json_file)
                
                
def paramaterize_to_csv(param_funcs, mask_dir, country1, country2, name='paramatized.csv', data_file=None):
    '''
        For a directory of images and masks, we iterate over each image/mask and apply 
        paramatizations. We then save the results to a csv file.
        
        param_funcs: a list of python functions that apply paramatizations to masks. [func1, func2, ...]
        mask_dir: a directory that contains sub-directories with images, masks, and json files. 
            If the masks do not exist, they will be created.
                format: mask_dir ------> file1 ----> file1.jpeg, file1.json, mask.png
                                 |
                                 ------> file2 ----> file2.jpeg, file2.json, mask.png
                                 ...
        country1: name of one of the border countries. Fully capitalized or 3 letter abbr.
        country2: name of one of the border countries. Fully capitalized or 3 letter abbr.
        name (optional): The name of the completed csv file.
        data_file (optional): a .shp or .kml file containing the border data needed to make masks. 
            Not necessary if masks already exist.
                          
    '''
    if not name.endswith('.csv'):
        name = name + '.csv'
    with open(name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        i = 1
        try:
            for folder in os.listdir(mask_dir):
                mask = None
                img = None
                try:
                    mask = imageio.imread(Path(mask_dir, folder, 'mask.png'))
                except Exception as e:
                    json_file = Path(mask_dir, folder, folder + ".json")
                    if os.path.exists(json_file) and data_file:
                        with open(json_file) as f:
                            meta = json.load(f)
                            bb = meta["resourceSets"][0]["resources"][0]["bbox"]
                        
                        img = imageio.imread(Path(mask_dir, folder, folder + '.jpeg'))

                        mask = make_mask(img, bb, data_file, country1, country2, False)
                    else:
                        print('No json at:', json_file)
                        
                if mask is not None:
                    results = []
                    for f in param_funcs:
                        try:
                            results.append(f(mask))
                        except Exception as e:
                            print(e)
                    writer.writerow([folder] + results)
                    print('#',i,':',results)
        except Exception as e:
            print(e)
        i += 1

    
if __name__ == "__main__":
    DATASET_FOLDER = "global-scrape" # name of folder containing dataset in /border-legibility/bing_maps/
    
    # Command line arguement to set scrape directory
    if len(sys.argv) > 1:
        DATASET_FOLDER = sys.argv[1]
    
    # Set to false in order to only go through one country1-country2 folder
    entire_directory = True
    dataset_path = f"../bing_maps/{DATASET_FOLDER}/" # directory of scrape
    folder = 'ang-nam' # only needs to be set if entire_directory == False

    # Creates .npy file for every tile in every country1-country2 folder in dataset_path
    try:
        for file in os.listdir(dataset_path):
            d = os.path.join(dataset_path, file)
            if os.path.isdir(d):
                segment_folder(dataset_path, file, dims=(256,256), shp_file=SHP_FILE)
            else:
                segment_folder(dataset_path, folder, dims=(256,256), shp_file=SHP_FILE)
    except:
        print("Error: directory (" + DATASET_FOLDER + ") does not exist.")

