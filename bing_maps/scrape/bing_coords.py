import numpy as np
import math
import json
import sys
from haversine import haversine, Unit

'''
Takes Bing Maps tiles and associated metadata to extract the lon/lat coordinates of
the corner points of the tile.
'''

def extractQuadkey(path):
    """ Pass in the path to a Bing Maps .json metadata file and get back
        the n-digit quadkey for the corresponding tile.
            Input:  path; a string and the path to the .json file we want to manipulate
            Output: key; the quadkey we seek
    """
    #Open the .json
    f = open(path, 'r')
    #Extract the image url
    url = json.load(f)['resourceSets'][0]['resources'][0]['imageUrl']
    #Steal the quadkey from the imageUrl
    key = url.split('/')[4].split('.')[0].replace('a', '')
    #Return it
    return key

def analyzeQuadkey(key):
    """ Pass in a quadkey and get back a list of the 4 corner points in GPS coords
        of the associated tile.
            Input:  key; a Bing Maps quadkey
            Output: corners; a list of GPS (lat,lon) corner points
    """
    #Zoom level == len(quadkey)
    level = len(key)
    
    print("Level: ", level)
    
    #This gives the Global (NOT individual tile) coordinate max for the zoom level
    coordMax = 256 * (2**level) - 1
    
    tileX, tileY = quadkeyToTileXY(key)
    
    #tileX != pixelX so we have to move into pixel space by reversing:
    #tileX = floor(pixelX / 256)
    #tileY = floor(pixelY / 256)
    
    pixelX = tileX * 256
    pixelY = tileY * 256
    #Verify safe bounding
    assert tileX <= coordMax
    assert tileY <= coordMax
    
    #For the sake of sanity, corners are ordered as follows:
    # c0 = top left, c1 = top right, c2 = bottom left, c3 = bottom right
    
    #This *should* give us the upper left corner point, then we can infer the others from there
    #Reverse engineer the (lat,lon):
    #pixelX = ((longitude + 180) / 360) * 256 * 2**level
    
    c0Lon = ((pixelX / (256 * 2**level)) * 360) - 180
    temp = ((pixelY / (256 * 2**level)) - 0.5) * (4 * np.pi)
    c0Lat = np.arcsin(-(1 - np.e**-temp)/(1 + np.e**-temp)) * 180 / np.pi #Drop the 2*pi*n since we only want n=0
    
    print("Upper Left: ", c0Lat, c0Lon)
    
    dA = (np.cos(c0Lat * (np.pi/180)) / 2**level) * 360
    print("dA: ", dA)
    
    c0 = (c0Lat, c0Lon)
    c1 = (c0Lat, c0Lon + dA) # longitude slightly off
    c2 = (c0Lat - dA, c0Lon)
    c3 = (c0Lat - dA, c0Lon + dA) #longitude slightly off
    
    return [c0, c1, c2, c3]
    
def quadkeyToTileXY(key):
    """ This method is from:
            https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
        Original Author: Joe Schwartz
        Translated to Python by Andrew Dunn
            Input:  key; a string representing the Bing Maps tile quadkey
            Output: tileX; the pixel X coordinate for the given tile
                    tileY; the pixel Y coordinate for the given tile
    """
    tileX = tileY = 0
    level = len(key)
    #Loop it, descending style
    for i in range(level, 0, -1):
        mask = 1 << (i - 1)
        #I haven't upgraded to the fancy Python 3.10 yet
        #so this is happening with dirty if/elif cases instead of
        #match...case
        #cast it as an int so we can do direct comparison
        digit = int(key[level - i])
        
        if digit == 0:
            continue
        
        elif digit == 1:
            tileX |= mask
        
        elif digit == 2:
            tileY |= mask
            
        elif digit == 3:
            tileX |= mask
            tileY |= mask
            
        else:
            raise Exception("Invalid quadkey!")
        
    #Return the coords
    return tileX, tileY

def getBBox(path, coordFormat=True):
    """ Give this the path to a .json file for the new scraping format and it returns
    the bounding box coordinates for the corners as defined by analyzeQuadkey()
        Input:  path; the filepath (as a string) to a .json file
        Output: box; a list of (lat, lon) bounding box coords just like analyzeQuadkey()
    """
    key = extractQuadkey(path)
    box = analyzeQuadkey(key)
    
    if not coordFormat:
        l = min([p[1] for p in box])
        b = min([p[0] for p in box])
        r = max([p[1] for p in box])
        t = max([p[0] for p in box])
        box = [b, l, t, r]

    return box

if __name__ == '__main__':
    pass