from genImgProducts import *
from getTimexShoreline import *
from datetime import datetime, timedelta
import requests
import cv2
import numpy as np
import os
import math
from io import BytesIO
import time

# get webcoos_token from the environment variable

webcoos_token = os.getenv('WebCOOS')
# webcoos_token = "OR DIRECTLY ADD YOUR TOKEN HERE"

def handle_time(time_str):
    """Convert a string to a datetime object."""
    if time_str is None:
        return None

    date = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    # Format the date in a readable format
    readable_date = date.strftime("%A, %B %d, %Y, %H:%M:%S")
    print(f"Readable date: {readable_date}")

    current_time = datetime.now()
    if date > current_time:
        print(f"Error: The requested time {date.strftime('%Y-%m-%d %H:%M:%S')} is in the future.")
        print(f"Current time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return

    return date

def get_inventory(start_time, end_time=None, api_token=webcoos_token, station='oakisland_west'):
    start_time = handle_time(start_time)
    end_time = handle_time(end_time)

    if api_token == '':
        raise ValueError('API token is required.')

    headers = {'Authorization': f'Token {api_token}', 'Accept': 'application/json'}
    endpoint_url = 'https://app.webcoos.org/webcoos/api/v1/elements/'
    # station = 'oakisland_west'
    inventory_url = f'https://app.webcoos.org/webcoos/api/v1/services/{station}-one-minute-stills-s3/inventory/'

    all_results = []
    current_start_time = start_time

    while True:
        # Set parameters for element request.
        params = {
            'service': station + '-video-archive-s3',
            # 'service': station + '-one-minute-stills-s3',
            'starting_after': current_start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'starting_before': end_time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        print(f"Request params: {params}")
        element_response = requests.get(endpoint_url, headers=headers, params=params)
        elements_data = element_response.json()

        results = elements_data['results']
        if not results:
            break

        all_results.extend(results)

        # Check if the results list is less than 100, meaning it's the last page
        if len(results) < 100:
            break

        # Update the current_start_time to the last temporal_min in the current results
        current_start_time_str = results[-1]['data']['extents']['temporal']['min']
        current_start_time = datetime.strptime(current_start_time_str, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        print(f"Updated current_start_time: {current_start_time}")

        # Ensure the new start time does not exceed the end time
        if current_start_time >= end_time:
            break

    # PRINT URL and TEMPORAL MIN only.
    for element in all_results:
        url = element['data']['properties']['url']
        temporal_min = element['data']['extents']['temporal']['min']
        # print(f"URL: {url}, Temporal Min: {temporal_min}")
    
    return all_results



# This function has been rebuilt and integrated into image_products.py
# The image_products.py file methods have no been integrated with multiprocessing_test.py
def download_and_process_video(mp4_url, stationName='oakisland_west'):
    start_time = time.time()  # Start the timer
    
    response = requests.get(mp4_url, stream=True)
    if response.status_code == 200:
        video_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            video_data.write(chunk)
        
        video_data.seek(0)  # Reset the file pointer to the beginning
        
        # Save video data to a temporary file-like object
        video_name = mp4_url.split('/')[-1].replace('.mp4', '')
        temp_video_path = f'{video_name}.mp4'
        with open(temp_video_path, 'wb') as f:
            f.write(video_data.getbuffer())
        
        download_end_time = time.time()
        download_duration = download_end_time - start_time
        print(f"Video download complete. Duration: {download_duration:.2f} seconds.")
        
        # Process the video
        photoAvg, timexName, photoBrt = genImgProducts(temp_video_path, stationName)
        
        
        ## Process Timex Image into average and brightest pixel images
        print(f"Timex image path: {timexName}")

        tranSL, fig_tranSL = getTimexShoreline(stationName, timexName)

        # Clean up temporary video file
        os.remove(temp_video_path)
        
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"Function execution complete. Total duration: {total_duration:.2f} seconds.")
        
        return photoAvg, timexName, photoBrt, tranSL, fig_tranSL
    else:
        print(f"Failed to download video: {response.status_code}")
        return None, None, None, None, None