import geopandas as gpd
import pandas as pd
import numpy as np
import math
import sqlite3
import folium
import fiona
import sys
import requests
import subprocess
import os
import shutil
import json
from shapely.geometry import MultiPolygon, Polygon, LineString, MultiLineString
from haversine import haversine, Unit
import time
from vincenty import vincenty

from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84

"""
    Original Author: Andrew Dunn
    Date:   07/10/2021
    Updated: 01/05/2023 by Skyler Crane
    Assumptions are made about the presence of key files:
        1. bing_key.txt
        2. cow-country-code.csv
        3. Borders2 shapefile
"""
#####################################
####### GLOBAL VARS GO HERE #########
#####################################

global callCount # Number of API calls made today
global countLimit # Number of API calls allowed on a given day
global startTime # What time scraping started today, use time.localtime() to set plz
global targetSideLength # Hoped for side length of a tile in km
global segmentFlag # Boolean to show if a particular border has a poorly spaced segment
global bordFile
global cowFile
global keyFile
global bingKey

###########################################################################
################# HELPER METHODS UP TOP, MAIN LOOP DOWN LOW ###############
###########################################################################
""" Handles file initialization before moving to other directories
"""
def filesInit(bPath, cPath, kPath):
    global bordFile
    bordFile = gpd.read_file(bPath)

    global cowFile
    cowFile = pd.read_csv(cPath)

    global keyFile
    keyFile = open(kPath, 'r')

    global bingkey
    bingKey = fetch_key()

def get_shape(in1, in2):
    """
    Input:
        in1, in2: two strings of country names that share a border
    Output:
        line: either a LineString object or a MultiLineString object
    """
    global bordFile
    borderFile = bordFile
    c1 = getFID(in1)
    c2 = getFID(in2)
    line = -1
    for i in range(len(borderFile)):
        if borderFile['LEFT_FID'][i] == c1:
            if borderFile['RIGHT_FID'][i] == c2:
                line = i
        if borderFile['LEFT_FID'][i] == c2:
            if borderFile['RIGHT_FID'][i] == c1:
                line = i
    if line < 0:
        raise Exception("Could not find border line!")

    return borderFile['geometry'][line]

def get_shape_line(in1, in2):
    """
    Input:
        in1, in2: two strings of country names that share a border
    Output:
        line: either a LineString object or a MultiLineString object
    """
    global bordFile
    borderFile = bordFile
    c1 = getFID(in1)
    c2 = getFID(in2)
    line = -1
    for i in range(len(borderFile)):
        if borderFile['LEFT_FID'][i] == c1:
            if borderFile['RIGHT_FID'][i] == c2:
                line = i
        if borderFile['LEFT_FID'][i] == c2:
            if borderFile['RIGHT_FID'][i] == c1:
                line = i
    if line < 0:
        raise Exception("Could not find border line!")
    print("Border found at line: " + str(line))
    return line

def get_shape_from_num(num):
    """
    Input:
        num: a non-negative integer, less than or equal to 319 for Borders2
    Output:
        line: either a LineString object or a MultiLineString object
    """
    global bordFile
    borderFile = bordFile

    if num < 0 or num >= len(borderFile):
        raise Exception("Could not find border line!")
    return borderFile['geometry'][num]

def getFID(name):
    """ Takes a country name for input and gives back the
        country code from Corellates of War CSV. """
    #split the handle and make them caps for cheching the COW
    #left, right, idx = handle.upper().split('-')
    name = name.strip().replace(' ', '').lower()
    #Load the csv and query
    global cowFile
    moo = cowFile
    col = moo['StateNme']
    code = moo['CCode']
    for i in range(len(col)):
        row = col[i].strip().replace(' ', '').lower()
        if row == name:
            return code[i]

    return -1

""" shapeLine is an int"""
def getShapeFIDs(shapeLine):
    global bordFile
    borderFile = bordFile

    return borderFile['LEFT_FID'][shapeLine], borderFile['RIGHT_FID'][shapeLine]

def getAbbrFromName(name):
    name = name.strip().replace(' ', '').lower()
    #Load the csv and query
    global cowFile
    moo = cowFile
    col = moo['StateNme']
    abb = moo['StateAbb']
    for i in range(len(col)):
        row = col[i].strip().replace(' ', '').lower()
        if row == name:
            return abb[i].lower()
    #Didn't find it, so it returns ???
    return "???"

def getNameFromAbbr(abbr):
    abbr = abbr.lower()
    #Load the csv and query
    global cowFile
    moo = cowFile
    col = moo['StateAbb']
    nme = moo['StateNme']
    for i in range(len(col)):
        row = col[i].strip().replace(' ', '').lower()
        if row == abbr:
            return nme[i]
    #Didn't find it, so it returns ???
    return "???"

def getAbbrFromCCode(code):
    #Load the csv and query
    global cowFile
    moo = cowFile
    col = moo['CCode']
    abb = moo['StateAbb']
    for i in range(len(col)):
        row = col[i]
        if row == code:
            return abb[i].lower()
    #Didn't find it, so it returns ???
    return "???"


def calcZoom(lat):
    """ Takes a latitude (float) as an input and calculates the correct
        Bing Maps zoom level to scrape for to get the most accurate tile size
        Formula outlined at:
            https://docs.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution
    """
    global targetSideLength
    target = targetSideLength #target side length of a tile in kilometers. 0.4 km ~ 0.25 miles
    mPerPixel = 156543.04 #Number of meters per pixel (from Bing docs)
    bestZoom = 0
    bestVal = 0
    for i in range(1,20):
        mapRes = mPerPixel * np.cos((lat*(np.pi/180))) / (2**i)
        check = mapRes * 256 /1000 #gets us into kms per tile side
        if((check-target)**2 < (bestVal-target)**2):
            bestZoom = i
            bestVal = check
    return bestZoom, bestVal


""" This method operates on a single LineString, use it to collect a series of points to scrape.
        input:  line, a LineString
        output: pts, an array of (long, lat) points as tuples
"""
def collectPoints(line):

    #Init output
    pts = []

    #Split the LineString into an array of coordinate pairs
    lineList = list(line.coords)
    for i in range(len(lineList)):
        pts.append(lineList[i])
    return pts


""" Takes an array of (long, lat) tuples and makes it into a new array
    of (lat, long) tupels that Bing Maps works with.
"""
def longToLat(pts):
    stp = []
    for i in range(len(pts)):
        lon = pts[i][0]
        lat = pts[i][1]
        stp.append((lat,lon))

    return stp

""" This one takes a list of line segment coords in (lat, long) form and creates a list of
    points to scrape that are spaced so that c[0] -> c[1] has a Haversine distance of
    targetSideLength. This will likely result in overlap between adjacent tiles, but this
    will reduce the amount of border missed if a tile has a border coord that is near an edge.
        input:  pointList, a list of (lat, lon) tuples
        output: scrapeList, a list of (lat, lon) tuples
"""
def generateScrapePoints(pointList):
    #Get global vars
    global targetSideLength
    target = targetSideLength * 1000 # Converts km to meters
    
    #Initialize Output
    scrape = []
    
    #Set up walk
    current = pointList[0]
    ender = pointList[len(pointList) - 1] #This is used for comparison
    
    #Begin walking input
    for i in range(1, len(pointList)):
        next = pointList[i]

        g = geod.Inverse(current[0], current[1], next[0], next[1])
        gap = g['s12'] # Distance from 'current' point to 'next' point (in meters)
       
        #Decide if incrementing
        
        if gap < target:
            # If the distance between the points is less than target distance,
            # Decide whether to scrape
            
            scrape = decide_append(scrape, pointList, current, next, gap, i)

            if i == (len(pointList) - 1):
                #Catch the end
                scrape.append(pointList[len(pointList) - 1])
        else:
            # Scrapes more tiles
            l = geod.InverseLine(current[0], current[1], next[0], next[1])
            ds = 0.4e3
            n = int(math.ceil(l.s13 / ds))
            
            for z in range(n):
                s = min(ds * z, l.s13)
                p = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                scrape.append((p['lat2'], p['lon2']))
                
            if i == (len(pointList) - 1):
                #Catch the end
                scrape.append(pointList[len(pointList) - 1])
        #Outside if/else
        current = next

    return scrape

""" Ensures that tiles which are within (targetSideLength) kilometers 
    One another are not scraped. Ensures that each tile (except perhaps 
    the first and second) contain different areas of land.
"""
def decide_append(scrape, points, current, next, gap, index):
    #This decides if the current spot should get appended based on distance
    #Look at the last one and the next one, decide if this one should get added
    if len(scrape) < 2:
        scrape.append(current)
        return scrape
    
    predecessor = scrape[len(scrape)-1]
    global targetSideLength
    goal = targetSideLength

    # gets distance between the two tiles
    back_gap = vincenty(predecessor, current)

    # If distance is at least desired tile length, add the tile to scrape
    if back_gap >= goal:
        scrape.append(current)
        return scrape
    
    return scrape


""" This returns a boolean regarding the truth of whether all points in the input list points
    lie on the line segment also passed in.
        Input:  segment- a list of coords from the shapefile
                points - a list of points intended to scrape
"""
def pointsOnSegment(segment, points):
    #Loop the points and test to see if they fall on a line segment
    delta = 1e-5
    for i in range(len(points)):
        pt = points[i]
        start = segment[0]
        collinear = False
        for j in range(1,len(segment)):
            end = segment[j]
            #Use collinearity to make sure we're good
            a,b = start
            m,n = pt
            x,y = end
            q1 = (n - b)*(x - m)
            q2 = (y - n)*(m - a)
            if (q1 - q2) < delta:
                collinear = True
            start = end
        if not collinear:
            print(np.linalg.det(test))
            print(i)
            print(pt)
            return False

    return True


""" takes a LineString or MultiLineString and rectifies each individual
    LineString into a scrapeable list of points.
        input: lineObj, a LineString or MultiLineString
        ouput: pointList, a list of (lat, lon) tupels
"""
def shapeLineToScrape(lineObj):
    try:
        cat = lineObj[0]
        multiLine = True
    except TypeError:
        multiLine = False
    pointList = []
    if(multiLine):
        for i in range(len(lineObj)):
            points = collectPoints(lineObj[i])
            points = longToLat(points)
            pts = generateScrapePoints(points)
            #Check the points for good spacing
            #validatePoints(pts)
            for j in range(len(pts)):
                pointList.append(pts[j])
    else:
        points = collectPoints(lineObj)
        points = longToLat(points)
        pts = generateScrapePoints(points)
        #Check the points for good spacing
        #validatePoints(pts)
        for j in range(len(pts)):
                pointList.append(pts[j])

    return pointList

### Now we have the list of points built, time to handle the scraping, saving, and directory

def createSubDirectory(parent, num):
    #Had to go to five digits of numbers based on 30k points in usa-can
    num = str(num).zfill(5)

    #Make and enter directory for this border
    outerFold = parent + '-' + num
    subprocess.call(["mkdir", outerFold])

    return outerFold

def createBorderDirectory(country1, country2):
    #Get the abbreviated names
    c1 = getAbbrFromCCode(country1)
    c2 = getAbbrFromCCode(country2)

    handle = c1 + "-" + c2

    subprocess.call(["mkdir", handle])

    return handle

def fetch_key():
    """
    You give fetch_key a file location and it gives you back
    the Bing Maps API key stored in a .txt file at that location.
    Input:
        location: a string containing the location of a .txt file
    Output:
        key: the Bing Maps API key stored in the file passed in
    """
    global keyFile
    keyfile = keyFile
    key = keyfile.readline()
    
    global bingKey
    bingKey = key

    return key

""" Things will not go well if you don't pass in a tuple of (lat, lon) coords in
    decimal form. Very particular. Also, only call this method inside the directory
    where you would like the data to be stored. Saves out a jpeg and a json.
"""
def scrape(point, subfolder):
    #Here is the place where you can change the url to the API:
    head = 'https://dev.virtualearth.net/REST/v1/Imagery/Metadata/Aerial/'
    mid  = '?&zoomLevel='
    tail = '&key='
    #Get the API key:
    global bingKey
    key = bingKey

    if len(key) == 0:
        raise Exception("Missing key")

    #Separate the lat and lon:
    callLat = point[0]
    callLong = point[1]

    #determine the zoom level for the url:
    zoom, _ = calcZoom(callLat)

    #Build the metadata url:
    url1 = head + str(callLat) + ',' + str(callLong) + mid + str(zoom) + tail + key
    #Leaving this as an empty string so error handling doesn't break
    url2 = ""

    #File names:
    header1 = subfolder + ".json"
    header2 = subfolder + ".jpeg"
    header3 = subfolder + ".txt"


    #Now the fancy part happens
    try:
        #Get the .json metadata:
        response1 = requests.get(url1, timeout=8)
        response1.raise_for_status()
        #Write the metadata to a file
        file1 = open(header1, "wb")
        file1.write(response1.content)
        file1.close()
        #Read the metadata and get the url2
        url2 = metaFetchURL(header1)
        #Get the jpeg
        response2 = requests.get(url2, timeout=8)
        response2.raise_for_status()
        #Write the jpeg
        file2 = open(header2, "wb")
        file2.write(response2.content)
        file2.close()
        
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
        fileE = open('ErrorH.txt', "w")
        fileE.write(url1 + '\n' + url2 + '\n' + str(errh))
        response1 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        response2 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        header1 = "cat1.jpeg"
        header2 = "cat2.jpeg"
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
        fileE = open('ErrorT.txt', "w")
        fileE.write(url1 + '\n' + url2 + '\n' + str(errt))
        response1 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        response2 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        header1 = "cat1.jpeg"
        header2 = "cat2.jpeg"
    except requests.exceptions.RequestException as err:
        print ("Uh oh, something else went wrong",err)
        fileE = open('ErrorG.txt', "w")
        fileE.write(url1 + '\n' + url2 + '\n' + str(err))
        response1 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        response2 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        header1 = "cat1.jpeg"
        header2 = "cat2.jpeg"
    except requests.exceptions.ReadTimeout as errT:
        print("A wild ReadTimeout appears! ", errT)
        fileE = open('ErrorRT.txt', "w")
        fileE.write(url1 + '\n' + url2 + '\n' + str(errT))
        response1 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        response2 = requests.get('http://www.thiscatdoesnotexist.com', timeout=5)
        header1 = "cat1.jpeg"
        header2 = "cat2.jpeg"

    #Not actually a requests call, so it's safe
    response3 = str(callLat) + "," + str(callLong) + "\n"
    response3 += url1 + "\n"
    response3 += url2 + "\n"
    response3 += "zoom:\t" + str(zoom)

    #Create log file
    file3 = open(header3, "w")
    file3.write(response3)
    ##Collect URLs in case of bad status code to easily manually scrape
    if response1.status_code != 200 or response2.status_code != 200:
        file3.write("Something went wrong with the scrape here :/")
    file3.close()
    #And we're done
    return

""" Takes a .json filename as input, opens the file, and gets the Bing
    Maps tile URL out of it and strips all forward slashes from it.
"""
def metaFetchURL(filename):
    #Open the file
    file = open(filename, 'r')
    md = json.load(file)
    #Hand back the image url for the tile we seek
    return md['resourceSets'][0]['resources'][0]['imageUrl']

""" Makes sure the current callCount is within the per-day limits
    and if not, puts the cycle to sleep for enough time to reset
    the API call counter.
"""
def checkCount():
    global callCount
    global countLimit

    if callCount >= countLimit:
        #Time to stop for the day, check the start time
        global startTime
        now = time.time()
        #Compare difference
        elapsed = math.ceil(now - startTime)
        napLength = 86460 - elapsed #One day of seconds plus a minute
        #Announce nap and snooze
        print('Sleeping for ' + str(napLength) + 'seconds')
        print('Current time is '+ time.strftime("%a, %d, %b, %Y, %H:%M:%S", time.localtime()))
        print('Nap ends at ' + time.strftime("%a, %d, %b, %Y, %H:%M:%S", time.localtime(time.time() + napLength)))

        time.sleep(napLength)
        print('Waking up!\n Current time is '+ time.strftime("%a, %d, %b, %Y, %H:%M:%S", time.localtime()))
        callCount = 0
        startTime = time.time()
    return



""" Takes an int as input, then handles everything to scrape the border
    corresponding to the line of the shapefile given by the input.
    Assumes that the pwd is inside where you want the data stored.
    (i.e. pwd is global-Borders2)
"""
def scrapeBorder(shapeLine):
    #Get the global vars
    global callCount
    global countLimit

    #Get the FID/CCodes
    cc1, cc2 = getShapeFIDs(shapeLine)

    #Make a parent directory for the border
    parent = createBorderDirectory(cc1, cc2)
    #Move into the new directory
    os.chdir(parent)

    #Get the geometry from the shapefile
    line = get_shape_from_num(shapeLine)
    #Assemble the points to scrape
    points = shapeLineToScrape(line)

    #Loop over all of the points
    for i in range(len(points)):
        #Make sure we can make the API call
        checkCount()

        #Make a directory and step in
        subHandle = createSubDirectory(parent, i)
        os.chdir(subHandle)

        #Scrape and save
        scrape(points[i], subHandle)

        #Increment callCount
        callCount += 1

        #Step out
        os.chdir("..")

    #Add the text file
    text = getNameFromAbbr(parent[:3]).replace(' ', '')
    text += "-"
    text += getNameFromAbbr(parent[4:]).replace(' ', '')
    text += '\t'

    global segmentFlag
    if segmentFlag:
        text += "\n Line segment fault: >5% distance variance found!"
        segmentFlag = False
    textFile = open(text.split('\t')[0]+'.txt', 'w')
    textFile.write(text)
    textFile.close()
    #Get back out of the directory
    os.chdir('..')


""" This sets up the global scraping run. Initializes the callCount
    and sets startTime (using time.mktime(time.localtime()) like
    we want)
    THIS IS WHERE YOU CAN CHANGE TILE TARGET SIZE!
        Input:
            skips, an optional array of ints for the shapefile lines to skip
"""
def globalScrape(skips=[], outerName=""):
    #Set global constants:
    bPath = 'Borders2/Int_Borders_version_2.shp'
    cPath = 'cow-country-code.csv'
    kPath = 'bing_key.txt'
    filesInit(bPath, cPath, kPath)

    global bordFile
    global cowFile
    global keyFile
    #First, targetSideLength
    global targetSideLength
    targetSideLength = 0.4 # Length of tile (kilometers)
    #Set up countLimit
    global countLimit
    countLimit = 48000
    #Set segmentFlag to False
    global segmentFlag
    segmentFlag = False
    #Set up callCount
    global callCount
    callCount = 0
    #Set up startTime
    global startTime
    startTime = time.time()

    #Assumed start point: /border-legibility/bing_maps/scrape
    os.chdir("..")
    #Make a top-level directory and step in
    if len(outerName) == 0:
        outerName = "global-no-stitch"
    subprocess.call(["mkdir", outerName])
    os.chdir(outerName)

    #Get the shapefile
    bf = bordFile
    #Loop over each line's geometry
    sum = 0
    for i in range(len(bf)):
        if(skips.count(i) > 0):
            #This line is in the skip list, so don't do it
            continue
        c1, c2 = getShapeFIDs(i)
        print(getNameFromAbbr(getAbbrFromCCode(c1)), ", ", getNameFromAbbr(getAbbrFromCCode(c2)), sep="")
        scrapeBorder(i)

def findShapeLine(c1, c2):
    bPath = 'Borders2/Int_Borders_version_2.shp'
    cPath = 'cow-country-code.csv'
    kPath = 'bing_key.txt'
    filesInit(bPath, cPath, kPath)
    #Assumes spelled names of c1, c2
    return get_shape_line(c1,c2)

def calculateScrapeDuration(size):
    global targetSideLength
    targetSideLength = size

    bf = gpd.read_file('Borders2/Int_Borders_version_2.shp')
    sumDist = 0
    sumPts  = 0

    for i in range(len(bf)):
        line = bf['geometry'][i]

        #Check if multiline
        try:
            line[0]
            ml = True
        except TypeError:
            ml = False

        if ml:
            #We have a multiline string, so do it that way
            for j in range(len(line)):
                ls = line[j]
                cor = ls.coords
                s = cor[0]
                for k in range(1,len(cor)):
                    e = cor[k]
                    sumDist += haversine(s,e, unit=Unit.KILOMETERS)
                    s = e
        else:
            cor = line.coords
            s = cor[0]
            for k in range(1,len(cor)):
                e = cor[k]
                sumDist += haversine(s,e, unit=Unit.KILOMETERS)
                s = e
        sumPts += len(shapeLineToScrape(line))
    sum = str(math.ceil(sumPts / 48000))
    print("At tile size " + str(targetSideLength) + " scrape will take " + sum + " days")
  
    

###############################################
############## MAIN METHOD ####################
###############################################
if __name__ == '__main__':
    
    """ Accepts command line args as follows:
        sys.argv[1] = "string-directory-name" (i.e. "global-Borders2")
    """
    # EXAMPLE: skips = [x for x in range(35)] #Set to skip the first 35 lines in the shapefile
    skips = []
    dirName = ""
    if len(sys.argv) ==1:
        dirName = 'global-scrape'
        globalScrape(skips, dirName)
        print("Done with global scrape! (Default directory name)")
    elif len(sys.argv) == 2:
        dirName = sys.argv[1]
        globalScrape(skips, dirName)
        print("Done with global scrape!")
    else:
        print("Too many command-line args given!")
