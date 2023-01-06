from glob import glob
import sys

'''
Author: Skyler Crane
Date: 13/30/2022

This code will create a list of all images in the scrape,
to be used by the Siamese model.

Input: command-line arg of name of scrape directory
Output: txt file will be placed in {SCRAPE_DIR}/tile-list.txt
'''

if len(sys.argv) > 1:
    SCRAPE_DIR = sys.argv[1]
else:
    SCRAPE_DIR = 'global-scrape'

tiles = []

## For every image folder, add to tiles list
for folder in glob( SCRAPE_DIR + '/*/*'):
    if folder[-1] == 't': # if 'folder' is a .txt file, skip
        pass
    else:
        tiles.append(folder)

with open( SCRAPE_DIR + "/tile-list.txt", mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(tiles))