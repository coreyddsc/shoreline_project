#genImgProducts.py
import os
import cv2
import math
import numpy as np

def getVidSnap(vid):
    """
    Create a snapshot of a video at a specific time.

    Args:
        vid (str): The path to the video file.

    Returns:
        snap (numpy.ndarray): The snapshot image from the video in RGB format.
    """
    # Create a VideoCapture object to read the video file
    vidObj = cv2.VideoCapture(vid)
    
    # Set the position of the video in milliseconds (5000 ms = 5 seconds)
    vidObj.set(cv2.CAP_PROP_POS_MSEC, 5000)
    
    # Read the frame at the specified time position
    success, snapBGR = vidObj.read()
    
    # Convert the frame from BGR (OpenCV default) to RGB color space
    snap = cv2.cvtColor(snapBGR, cv2.COLOR_BGR2RGB)
    
    # Return the snapshot image in RGB format
    return snap




def genImgProducts(vidName, stationName='oakisland_west'):
    # Check for and create the "images" directory
    base_dir = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Directory exists: {base_dir}")

    # Create station directory
    station_dir = os.path.join(base_dir, stationName)
    if not os.path.exists(station_dir):
        os.makedirs(station_dir)
        print(f"Created station directory: {station_dir}")
    else:
        print(f"Station directory exists: {station_dir}")

    # Create directories for image types
    timex_dir = os.path.join(station_dir, 'timex')
    brt_dir = os.path.join(station_dir, 'bright')
    avg_dir = os.path.join(station_dir, 'average')
    for dir_path in [timex_dir, brt_dir, avg_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory exists: {dir_path}")

    # Create file names.
    vidPath = os.path.join(os.getcwd(), vidName)
    if not os.path.isfile(vidPath):
        raise FileNotFoundError(f"Video file not found: {vidPath}")
    fname, _ = os.path.splitext(os.path.basename(vidPath))
    timexName = os.path.join(timex_dir, fname + '-timex.jpeg')
    brtName = os.path.join(brt_dir, fname + '-brt.jpeg')
    avgName = os.path.join(avg_dir, fname + '-avg.jpeg')

    # Set sub-sample interval. (1 frame every 10 secs).
    subSampleRate = 10

    # Open video.
    vidObj = cv2.VideoCapture(vidPath)
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    totFrameCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    vidLength = math.floor((totFrameCount / fps) - 1)

    # Target frame count for final product calculation.
    tgtFrameCount = math.floor(vidLength / subSampleRate)

    w = 1 / tgtFrameCount
    end = tgtFrameCount - 1

    snap = getVidSnap(vidPath)
    photoAvg = np.zeros(np.shape(snap))
    photoBrt = np.zeros(np.shape(snap))

    # Generate the time-averaged and brightest pixel image products.
    for i in range(end):
        timeSet = i * 1000 * subSampleRate
        vidObj.set(cv2.CAP_PROP_POS_MSEC, timeSet)
        success, snap = vidObj.read()

        snapd = np.array(snap).astype(float)
        if i == 0:
            photoAvg = w * snapd
            photoBrt = snapd
        else:
            photoAvg += w * snapd
            photoBrt = np.where(snapd > photoBrt, snapd, photoBrt)

    # Scale/save time-averaged image product.
    avgScale = photoAvg.max()
    photoAvg = np.uint8(254 * (photoAvg / avgScale))
    cv2.imwrite(timexName, photoAvg)

    # Scale/save brightest pixel image product.
    brtScale = photoBrt.max()
    photoBrt = np.uint8(254 * (photoBrt / brtScale))
    cv2.imwrite(brtName, photoBrt)

    # Convert to RGB.
    photoAvg = cv2.cvtColor(photoAvg, cv2.COLOR_BGR2RGB)
    photoBrt = cv2.cvtColor(photoBrt, cv2.COLOR_BGR2RGB)

    return photoAvg, timexName, photoBrt



