# getTimexShoreline.py
import cv2 
from datetime import datetime 
from itertools import chain
import json 
# import math 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from PIL import Image, ImageDraw 
import re 
import scipy.signal as signal 
from skimage.filters import threshold_otsu 
from skimage.measure import profile_line 
from statsmodels.nonparametric.kde import KDEUnivariate

def getStationInfo(ssPath):
    # Loads json and converts data to NumPy arrays.
    with open(ssPath, 'r') as setupFile:
        stationInfo = json.load(setupFile)
    stationInfo['Dune Line Info']['Dune Line Interpolation'] = np.asarray(stationInfo['Dune Line Info']['Dune Line Interpolation'])
    stationInfo['Shoreline Transects']['x'] = np.asarray(stationInfo['Shoreline Transects']['x'])
    stationInfo['Shoreline Transects']['y'] = np.asarray(stationInfo['Shoreline Transects']['y'])
    return stationInfo

def mapROI(stationInfo, photo):
    # Draws a mask on the region of interest and turns the other pixel values to nan.
    w, h = photo.shape[1], photo.shape[0]
    transects = stationInfo['Shoreline Transects']
    xt = np.asarray(transects['x'], dtype=int)
    yt = np.asarray(transects['y'], dtype=int) 
    cords = np.column_stack((xt[:, 1], yt[:, 1]))
    cords = np.vstack((cords, np.column_stack((xt[::-1, 0], yt[::-1, 0]))))
    cords = np.vstack((cords, cords[0]))  
    poly = list(chain.from_iterable(cords))
    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    mask = np.array(img)
    maskedImg = photo.astype(np.float64)
    maskedImg[mask == 0] = np.nan
    maskedImg /= 255
    return maskedImg

def improfile(rmb, stationInfo):
    # Extract intensity profiles along shoreline transects.
    transects = stationInfo['Shoreline Transects']
    xt = np.asarray(transects['x'])
    yt = np.asarray(transects['y'])
    n = len(xt)
    imProf = [profile_line(rmb, (yt[i,1], xt[i,1]), (yt[i,0], xt[i,0]), mode='constant') for i in range(int(2*n/3-1), int(2*n/3+1))]
    improfile = np.concatenate(imProf)[~np.isnan(np.concatenate(imProf))]
    return improfile

def ksdensity(P, **kwargs):
    # Univariate kernel density estimation.
    x_grid = np.linspace(P.max(), P.min(), 1000) # Could cache this.
    kde = KDEUnivariate(P)
    kde.fit(**kwargs)
    pdf = kde.evaluate(x_grid)
    return (pdf, x_grid)


def extract(stationInfo, rmb, maskedImg, threshInfo):
    # Uses otsu's threshold to find shoreline points based on water orientation.
    stationname = stationInfo['Station Name']
    slTransects = stationInfo['Shoreline Transects']
    dtInfo = stationInfo['Datetime Info']
    date = dtInfo.date()
    xt = np.asarray(slTransects['x'])
    yt = np.asarray(slTransects['y'])
    orn = stationInfo['Orientation']
    thresh = threshInfo['Thresh']
    thresh_otsu = threshInfo['Otsu Threshold']
    thresh_weightings = threshInfo['Threshold Weightings']
    length = min(len(xt), len(yt))
    trsct = range(0, length)
    values = [0]*length
    revValues = [0]*length
    yList = [0]*length
    xList = [0]*length

    def find_first_exceeding_index(values, threshold):
        values = np.array(values)
        for i in range(1, len(values)):
            if (values[i-1] < threshold and values[i] >= threshold) or (values[i-1] >= threshold and values[i] < threshold):
                return i
        return None

    if orn == 0:
        for i in trsct:
            x = int(xt[i][0])
            yMax = int(yt[i][1])
            yMin = int(yt[i][0])
            y = yMax - yMin
            yList[i] = np.zeros(shape=y)
            val = [0]*(yMax - yMin)
            for j in range(len(val)):
                k = yMin + j
                val[j] = rmb[k][x]
            val = np.array(val)
            values[i] = val

        idx = [0]*len(xt)
        xPt = [0]*len(xt)
        yPt = [0]*len(xt)
        for i in range(len(values)):
            idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            if idx[i] is None:
                yPt[i] = None
                xPt[i] = None
            else:
                yPt[i] = min(yt[i]) + idx[i]
                xPt[i] = int(xt[i][0])
        shoreline = np.vstack((xPt, yPt)).T
    else:
        for i in trsct:
            xMax = int(xt[i][0])
            y = int(yt[i][0])
            yList[i] = np.full(shape=xMax, fill_value=y)
            xList[i] = np.arange(xMax)
            values[i] = rmb[y][0:xMax]
            revValues[i] = rmb[y][::-1]

        idx = [0]*len(yt)
        xPt = [0]*len(yt)
        yPt = [0]*len(yt)
        for i in range(len(revValues)):
            idx[i] = find_first_exceeding_index(values[i], thresh_otsu)
            xPt[i] = idx[i]
            yPt[i] = int(yt[i][0])
        shoreline = np.vstack((xPt, yPt)).T

    # Convert numpy data types to native Python types and handle None values in shoreline
    slVars = {
        'Station Name': stationname,
        'Date': str(date),
        'Time Info': str(dtInfo),
        'Thresh': float(thresh),
        'Otsu Threshold': float(thresh_otsu),
        'Shoreline Transects': {
            'x': xt.tolist(),
            'y': yt.tolist()
        },
        'Threshold Weightings': [float(w) for w in thresh_weightings],
        'Shoreline Points': [[float(item) if item is not None else None for item in point] for point in shoreline]
    }

    try:
        del slVars['Time Info']['DateTime Object (UTC)']
        del slVars['Time Info']['DateTime Object (LT)']
    except:
        pass

    if isinstance(slVars['Shoreline Transects']['x'], np.ndarray):
        slVars['Shoreline Transects']['x'] = slVars['Shoreline Transects']['x'].tolist()
        slVars['Shoreline Transects']['y'] = slVars['Shoreline Transects']['y'].tolist()

    # Create directories if they do not exist
    base_dir = os.path.join(os.getcwd(), 'transect_jsons', stationname)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    # Save JSON file to the directory
    fname = os.path.join(base_dir, f'{stationname}-{datetime.strftime(dtInfo, "%Y-%m-%d_%H%M")}.avg.slVars.json')
    with open(fname, "w") as f:
        json.dump(slVars, f)
    print(f"Saved JSON to: {fname}")
    
    return shoreline


def pltFig_tranSL(stationInfo, photo, tranSL):
    # Print the dimensions of the photo
    print(f"Photo dimensions: {photo.shape}")

    # Print the first few shoreline coordinates
    print(f"Shoreline coordinates (first 10): {tranSL[:10]}")
    # Creates shoreline product.
    stationname = stationInfo['Station Name']
    dtInfo = stationInfo['Datetime Info']
    date = str(dtInfo.date())
    time = str(dtInfo.hour).zfill(2) + str(dtInfo.minute).zfill(2)  # Ensure two digits for hour and minute
    Di = stationInfo['Dune Line Info']
    duneInt = Di['Dune Line Interpolation']
    xi, py = duneInt[:,0], duneInt[:,1]
    plt.ioff()
    fig_tranSL = plt.figure()
    plt.imshow(photo, interpolation='nearest')
    plt.xlabel("Image Width (pixels)", fontsize=10)
    plt.ylabel("Image Height (pixels)", fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.plot(tranSL[:, 0], tranSL[:, 1], color='r', linewidth=2, label='Detected Shoreline')
    plt.plot(xi, py, color='blue', linewidth=2, label='Baseline', zorder=4)
    plt.title(('Transect Based Shoreline Detection (Time Averaged)\n' + stationname.capitalize() + 
               ' on ' + date + ' at ' + time[:2] + ':' + 
               time[2:] + ' UTC'), fontsize = 12)
    plt.legend(prop={'size': 9})
    plt.tight_layout()
    
    # Construct the save path for the figure
    base_dir = os.path.join(os.getcwd(), 'images', stationname, 'average')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    saveName = os.path.join(base_dir, f'{stationname}-{date}_{time}.tranSL-avg.fix.jpeg')
    plt.savefig(saveName, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"Saved fig_tranSL to: {saveName}")
    
    return fig_tranSL

def getTimexShoreline(stationName, imgName):
    # Main program.
    cwd = os.getcwd()
    stationPath = os.path.join(cwd, stationName + '.config.json')
    if not os.path.exists(stationPath):
        # Try the alternative path if the stationPath doesn't exist
        stationPath = os.path.join('c:\\Users\\Corey Dearing\\Desktop\\webCOOS\\webcoos_request', stationName + '.config.json')
    
    print(f"Station path: {stationPath}")
    print(f"Station path: {stationPath}")
    stationInfo = getStationInfo(stationPath) 
    dtObj = datetime.strptime(re.sub(r'\D', '', imgName), '%Y%m%d%H%M%S')
    stationInfo['Datetime Info'] = dtObj
    
    # Converts image color scale.
    photoAvg = cv2.cvtColor(cv2.imread(imgName), cv2.COLOR_BGR2RGB)
    new_size = (int(photoAvg.shape[1] * 0.3), int(photoAvg.shape[0] * 0.3))
    resized_image = cv2.resize(photoAvg, new_size, interpolation=cv2.INTER_AREA)
    
    # Creating an array version of image dimensions for plotting.
    h, w = resized_image.shape[:2]
    xgrid, ygrid = np.linspace(0, w, w, dtype=int), np.linspace(0, h, h, dtype=int)
    X, Y = np.meshgrid(xgrid, ygrid, indexing = 'xy')
    
    # Maps regions of interest on plot.
    maskedImg = mapROI(stationInfo, resized_image)
    
    # Computes rmb.
    rmb = maskedImg[:,:,0] - maskedImg[:,:,2]
    P = improfile(rmb, stationInfo).reshape(-1, 1)
    
    # Computing probability density function and finds threshold points.
    pdfVals, pdfLocs = ksdensity(P)
    thresh_weightings = [(1/3), (2/3)]
    peaks = signal.find_peaks(pdfVals)
    peakVals = np.asarray(pdfVals[peaks[0]])
    peakLocs = np.asarray(pdfLocs[peaks[0]])  

    thresh_otsu = threshold_otsu(P)
    I1 = np.asarray(np.where(peakLocs < thresh_otsu))
    J1, = np.where(peakVals[:] == np.max(peakVals[I1]))
    I2 = np.asarray(np.where(peakLocs > thresh_otsu))
    J2, = np.where(peakVals[:] == np.max(peakVals[I2]))
    thresh = (thresh_weightings[0]*peakLocs[J1] +
              thresh_weightings[1]*peakLocs[J2])
    thresh = float(thresh)
    threshInfo = {
        'Thresh':thresh, 
        'Otsu Threshold':thresh_otsu,
        'Threshold Weightings':thresh_weightings
        }

    # Generates final json and figure for shoreline products.
    tranSL = extract(stationInfo, rmb, maskedImg, threshInfo)
    fig_tranSL = pltFig_tranSL(stationInfo, resized_image, tranSL)
    
    return(tranSL, fig_tranSL)


###########################################################################


# stationName = 'currituck_hampton_inn'

# imgNames = ['timex.currituck_hampton_inn-2024-07-08-214121Z.jpg',
#             'timex.currituck_hampton_inn-2024-07-08-145109Z.jpg',
#             'timex.currituck_hampton_inn-2024-07-08-001249Z.jpg']

# for imgName in imgNames:
#     tranSL, fig_tranSL = getTimexShoreline(stationName, imgName)
