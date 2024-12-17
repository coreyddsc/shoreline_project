import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import cv2
from read_jsons import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GetImages:
    def __init__(self, station_name):
        self.station_name = station_name
        self.image_dir = f'../images/{station_name}/'
        self.image_list = os.listdir(self.image_dir)
        self.image_list.sort()
        
        
    def get_image_dir(self):
        return self.image_dir
    
    def set_image_type(self, image_type):
        self.image_type = image_type
        self.image_dir = f'../images/{self.station_name}/{image_type}/'
        return self.image_dir
    
    # get the head of this directory
    def get_image_list(self):
        self.image_list = os.listdir(self.image_dir)
        return self.image_list
    
    # parse file name in image list to get the datetime
    def parse_image_datetime(self):
        self.image_datetime = []
        for image in self.image_list:
            image_datetime = re.findall(r'\d{4}-\d{2}-\d{2}-\d{6}', image)
            # format the datetime to 'YYYY-MM-DD HH:MM:SS'
            image_datetime = [datetime.strptime(image_datetime[0], '%Y-%m-%d-%H%M%S').strftime('%Y-%m-%d %H:%M:%S')]
            self.image_datetime.append(image_datetime[0])
            
        self.image_list.sort()
        return self.image_datetime
    
    # create a dataframe with image list and image datetime
    def create_image_df(self):
        self.image_df = pd.DataFrame({'Image': self.image_list, 'Datetime': self.image_datetime})
        # set datetime as index
        self.image_df.set_index('Datetime', inplace=True)
        return self.image_df
    
    # use index from image_df to get a list of dates in a user defined range
    def get_date_range(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = self.image_df.loc[start_date:end_date].index
        return self.date_range
    
    # use the image/datetime df to query the directory for images between two dates
    # by first querying the datetime index.
    # take a list of dates to subset image_df
    def get_image_range(self, date_range):
        self.image_range = self.image_df.loc[date_range]
        return self.image_range
    
    # get and store the image using cv2 imread to read the image by image directory and name
    # store the image in a dictionary with the datetime as the key
    # Read and store images using cv2.imread, with the index as the key
    def get_images(self):
        self.images = {}
        for idx, row in self.image_range.iterrows():
            image_path = os.path.join(self.image_dir, row['Image'])  # Use os.path.join for file paths
            self.images[idx] = cv2.imread(image_path)  # Store image with index as the key
        return self.images
    
    # resize the images in the dictionary and store them in a new dictionary
    def resize_images(self, scale_percent=100):
        self.scaled_images = {}
        for idx, row in self.image_range.iterrows():
            image = self.images[idx]
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            self.scaled_images[idx] = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return self.scaled_images
    
    # create rmb images from the dictionary of images with standard or scaled as an option
    def norm_rmb_images(self, scale=False):
        self.norm_rmb = {}
        self.norm_bmr = {}
        if scale:
            images = self.scaled_images
        else:
            images = self.images
        
        for idx, row in images.items():
            image = images[idx]
            blue_channel = image[:, :, 0]
            red_channel = image[:, :, 2]
            
            # Compute the red-minus-blue difference image
            red_minus_blue = red_channel.astype(int) - blue_channel.astype(int)
            blue_minus_red = blue_channel.astype(int) - red_channel.astype(int)
            
            # Normalize the red-minus-blue image to the range [0, 255]
            rmb_normalized = np.clip(red_minus_blue, 0, 255).astype(np.uint8)
            bmr_normalized = np.clip(blue_minus_red, 0, 255).astype(np.uint8)
            
            self.norm_rmb[idx] = rmb_normalized
            self.norm_bmr[idx] = bmr_normalized
            
        # return self.norm_rmb
    
    
    # get otsu threshold from norm_rmb images
    def otsu_threshold(self):
        self.otsu_norm = None
        self.thresholds = {}
        # if station name is currituck_hampton_inn, oakisland_west, use rmb images
        # for idx, row in self.norm_rmb.items():
        # update otsu method to choose the filter based on station name
        # use bmr images for jennette location
        # if station name is jennette, use bmr images
        if self.station_name == 'jennette_north':
            self.otsu_norm = self.norm_bmr
        # else station name is in list ['currituck_hampton_inn', 'oakisland_west']
        elif self.station_name in ['currituck_hampton_inn', 'oakisland_west']:
            self.otsu_norm = self.norm_rmb
        else:
            print("Station name not recognized. Please check the station name.")
            print("Using Blue-Minus-Red images for Otsu thresholding.")
            self.otsu_norm = self.norm_bmr
            
        print(f"otsu_norm type: {type(self.otsu_norm)}")    
        for idx, row in self.otsu_norm.items():
            _, otsu_thresh = cv2.threshold(row, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.thresholds[idx] = otsu_thresh
        # return self.thresholds
    
    # get image gradient from input dictionary norm or otsu threshold 
    def image_gradient(self, dict_name = 'otsu'):
        self.image_gradients = {}
        if dict_name == 'otsu':
            images = self.thresholds
        else:
            images = self.norm_rmb
        
        for idx, row in images.items():
            image = images[idx]
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gradx = cv2.Sobel(row, cv2.CV_64F, 1, 0, ksize=5)
            grady = cv2.Sobel(row, cv2.CV_64F, 0, 1, ksize=5)
            gradient = np.sqrt(gradx**2 + grady**2)
            self.image_gradients[idx] = gradient
            
        # return self.image_gradients
        
    # create gradient mask from image gradient
    def gradient_mask(self, scale=False):
        self.gradient_masks = {}

        if scale:
            images = self.scaled_images
        else:
            images = self.images

        for idx, gradient in self.image_gradients.items():
            image_color = images[idx]  # Image is already in BGR format

            # Create a red mask where gradient > 0
            red_mask = np.zeros_like(image_color)  # Initialize the mask as a BGR image
            red_mask[gradient > 0] = [0, 0, 255]  # Set red where gradient is non-zero
            
            # Apply the red mask only at gradient locations, leave the rest unchanged
            combined_image = np.where(red_mask > 0, red_mask, image_color)
            
            # Store the combined image
            self.gradient_masks[idx] = combined_image

        # return self.gradient_masks

            
    # create image gradient video
    def gradient_video(self, output_video='gradient_video.mp4', fps=10):
        if len(self.image_gradients) == 0:
            print("No images to create video.")
            return
        
        # Get the dimensions of the first image to initialize the video writer
        first_image = next(iter(self.image_gradients.values()))
        height, width = first_image.shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        for idx in self.image_gradients:
            frame = self.image_gradients[idx]
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)
        
        video_writer.release()
        
        
    # create image gradient video
    def gradient_timelapse(self, output_video='gradient_video.mp4', fps=10):
        if len(self.image_gradients) == 0:
            print("No images to create video.")
            return

        # Get the dimensions of the first image to initialize the video writer
        first_image = next(iter(self.image_gradients.values()))
        height, width = first_image.shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        for idx in self.image_gradients:
            # Get the original image and its gradient mask
            original_image = self.images[idx]  # The original color image
            gradient = self.image_gradients[idx]  # The gradient for this image
            
            # Normalize the gradient and convert it to uint8 type
            gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert the gradient to a BGR image
            gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
            
            # Create the red mask where gradient > 0
            red_mask = np.zeros_like(gradient_bgr)
            red_mask[gradient > 0] = [0, 0, 255]  # Red color in BGR format
            
            # Overlay the red mask on the original image
            combined_frame = cv2.addWeighted(original_image, 1, red_mask, 1, 0)
            
            # timestamp = idx.strftime('%Y-%m-%d %H:%M:%S')
            timestamp = idx
            
            # cv2.putText(combined_frame, timestamp, (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(combined_frame, timestamp, (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
            
            # Write the frame to the video
            video_writer.write(combined_frame)
        
        video_writer.release()
        print(f"Gradient video saved as {output_video}")


    
    # Create video from the sequence of images, with an option to scale them and draw shorelines
    def create_video(self, shoreline_coords, output_video='output_video.mp4', fps=10, scale_percent=100):
        self.slc = shoreline_coords
        if len(self.images) == 0:
            print("No images to create video.")
            return
        
        # Get the dimensions of the first image to initialize the video writer
        first_image = next(iter(self.images.values()))
        height, width, _ = first_image.shape
        
        # Adjust width and height according to the scaling percentage
        scaled_width = int(width * scale_percent / 100)
        scaled_height = int(height * scale_percent / 100)
        size = (scaled_width, scaled_height)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4 format
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, size)

        # Get the colormap for the shorelines
        colors = cm.viridis(np.linspace(0, 1, len(self.slc)))

        for idx, row in self.image_range.iterrows():
            frame = self.images[idx]

            # Resize the frame according to the specified scale
            resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            
            # timestamp = idx.strftime('%Y-%m-%d %H:%M:%S')
            timestamp = idx
            
            # cv2.putText(resized_frame, timestamp, (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(resized_frame, timestamp, (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Select the corresponding shoreline based on the datetime index
            if idx in self.slc.index:  # Check if idx exists in the shoreline DataFrame index
                shoreline_points_x = self.slc.loc[idx].values  # Select shoreline by index
                shoreline_points_y = self.slc.columns.astype(float)

                # Convert points to integer pixel coordinates (without scaling)
                points = np.column_stack((shoreline_points_x, shoreline_points_y)).astype(np.int32)

                # Ensure that the points are reshaped appropriately for polylines
                points = points.reshape((-1, 1, 2))

                # Draw the shoreline on the frame
                color = tuple(map(int, (colors[self.slc.index.get_loc(idx)][:3] * 255)))  # Convert to BGR for cv2
                cv2.polylines(resized_frame, [points], isClosed=False, color=color, thickness=2)
                

            # Write the frame with the shoreline to the video
            video_writer.write(resized_frame)

        video_writer.release()
        print(f"Video with shorelines saved as {output_video}")