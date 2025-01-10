import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import cv2
# import all from read_jsons if in tools directory and tools.read_jsons otherwise
try:
    from .read_jsons import * # read_jsons.py
except:
    from read_jsons import * # read_jsons.py 
# from read_jsons import * # read_jsons.py

try:
    from .get_image_files import * # get_image_files.py
except:
    from get_image_files import * # get_image_files.py
# from get_image_files import * # get_image_files.py
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotly.graph_objects as go

class ImageAnalysis:
    def __init__(self, img_dict, image_df=None): # datetime, image file dictionary
        self.img_dict = img_dict
        if image_df is None:
            self.image_df = pd.DataFrame(index=pd.to_datetime([]))
        else:
            self.image_df = image_df
            
        # Set initial Region of Interest (ROI) for image analysis to full image dimensions
        first_image = img_dict[list(img_dict.keys())[0]]
        self.ROI = (0, 0, first_image.shape[1], first_image.shape[0])
        self.ROI_dict = None
        
            
    # image segmentation
    # set region of interest (ROI) for image analysis
    def set_ROI(self, x1, y1, x2, y2):
        self.ROI = (x1, y1, x2, y2)
        # trim image copy and store in new dictionary
        self.ROI_dict = {}
        for img_date in self.img_dict.keys():
            img = self.img_dict[img_date]
            self.ROI_dict[img_date] = img[y1:y2, x1:x2]
        
    def segment_images(self, grid_size=25):
        self.grid_size = grid_size
        """Segment images into grid regions and store in segmented_data"""
        if self.ROI_dict is None:
            self.set_ROI(*self.ROI)  # Ensure ROI is applied
        
        self.segmented_data = {}  # {date: {region_label: region_array}}
        
        for img_date, img in self.ROI_dict.items():
            regions = {}
            roi_height = img.shape[0]
            roi_width = img.shape[1]
            
            # Iterate through the grid
            for y in range(0, roi_height, grid_size):
                for x in range(0, roi_width, grid_size):
                    # Define region bounds
                    x_end = min(x + grid_size, roi_width)
                    y_end = min(y + grid_size, roi_height)
                    
                    # Extract region and assign label
                    region = img[y:y_end, x:x_end]
                    label = f"region_{y//grid_size}_{x//grid_size}"
                    regions[label] = region
            
            self.segmented_data[img_date] = regions
        
        # After segmentation, build the dataframe and store it
        self.build_segmented_dataframe()
        return self.segmented_data

    def build_segmented_dataframe(self):
        """Build a DataFrame from the segmented data and store it"""
        if self.segmented_data is None:
            raise ValueError("Segmented data not available. Please run segment_images() first.")
        
        rows = []
        
        # Loop through segmented data and create DataFrame rows
        for img_date, regions in self.segmented_data.items():
            for region_label, region_array in regions.items():
                rows.append({
                    'Date': img_date,
                    'Region': region_label,
                    'Region_Array': region_array
                })
        
        # Convert the rows to a DataFrame and store it in the class
        self.segmented_dataframe = pd.DataFrame(rows)
        
    def add_bgr_stats(self):
        """Add max, average, and median BGR values to the segmented dataframe"""
        if self.segmented_dataframe is None:
            raise ValueError("Segmented dataframe not available. Please run segment_images() first.")
        
        # Create new columns for max, avg, and median BGR values
        max_bgr = []
        avg_bgr = []
        median_bgr = []
        
        for _, row in self.segmented_dataframe.iterrows():
            region_array = row['Region_Array']
            
            # Calculate max, average, and median for each BGR channel
            max_bgr_value = tuple(np.max(region_array, axis=(0, 1)))  # Max BGR values across region
            avg_bgr_value = tuple(np.mean(region_array, axis=(0, 1)))  # Avg BGR values across region
            median_bgr_value = tuple(np.median(region_array, axis=(0, 1)))  # Median BGR values across region
            
            max_bgr.append(max_bgr_value)
            avg_bgr.append(avg_bgr_value)
            median_bgr.append(median_bgr_value)
        
        # Add these as new columns in the dataframe
        self.segmented_dataframe['Max_BGR'] = max_bgr
        self.segmented_dataframe['Avg_BGR'] = avg_bgr
        self.segmented_dataframe['Median_BGR'] = median_bgr
        
    def color_normalization(self, region, norm_type='l1', weights=(0.3, 0.4, 0.3)):
        """
        Normalize the color values of a region using a specified method.

        Parameters:
            region (numpy.ndarray): BGR region (H x W x 3).
            norm_type (str): Normalization method:
                - 'l1': Normalize by the Manhattan norm (sum of absolute values).
                - 'l1_w': Normalize by the Manhattan norm with weights (adaptive scaling).
                - 'l2': Normalize by the Euclidean norm (square root of sum of squares).
            weights (tuple): Weights for the 'l1_w' normalization (default=(0.3, 0.4, 0.3)).

        Returns:
            numpy.ndarray: Normalized region.
        """
        bgr = region.copy().astype(np.float32)
        
        # Compute the normalization divisor based on the specified method
        if norm_type is None:
            return bgr
        elif norm_type == 'l1':
            # Manhattan norm (sum of absolute values)
            divisor = np.sum(np.abs(bgr), axis=-1, keepdims=True)
        elif norm_type == 'l1_w':
            # Weighted Manhattan norm (adaptive scaling)
            divisor = (
                weights[0] * bgr[..., 0] +  # Weighted Blue
                weights[1] * bgr[..., 1] +  # Weighted Green
                weights[2] * bgr[..., 2]    # Weighted Red
            )[..., np.newaxis]
        elif norm_type == 'l2':
            # Euclidean norm (square root of sum of squares)
            divisor = np.sqrt(np.sum(bgr**2, axis=-1, keepdims=True))
        else:
            raise ValueError(f"Invalid norm_type: {norm_type}")
        
        # Avoid division by zero by adding a small constant to the divisor
        divisor = np.maximum(divisor, 1e-6)
        
        # Normalize the BGR values
        norm_bgr = bgr / divisor
        
        return norm_bgr

    
    def colorspace_transformation(self, region, transform_type='gray'):
        # Convert the region to grayscale
        if transform_type == 'gray':
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            return gray
        # HSV
        elif transform_type == 'hsv':
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            return hsv
        # LAB
        elif transform_type == 'lab':
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            return lab
        # YCrCb
        elif transform_type == 'ycrcb':
            ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
            return ycrcb
        else:
            raise ValueError(f"Invalid transform_type: {transform_type}")
        
    def pixel_sand(self, region, lambda_val=3):
        """Calculate the sand pixel value for a region based on the range of BGR values"""
        bgr = region.copy()
        # Compute the max and min values for each pixel across the last dimension
        max_vals = np.max(bgr, axis=-1)
        min_vals = np.min(bgr, axis=-1)
        # Compute the range for each pixel
        range_vals = max_vals - min_vals
        # Create a mask where the range is less than or equal to lambda
        final_mask = range_vals <= lambda_val
        # Count the number of sand pixels
        sand_pixels = np.sum(final_mask)
        # Print percentage of sand pixels
        print(f"Sand Pixels: {sand_pixels}")
        print(f"Percentage of Sand Pixels: {(sand_pixels / final_mask.size)*100:.2f}%")
        # print shape of final mask
        print(f"Shape of final mask: {final_mask.shape}")
        
        self.percent_sand = (sand_pixels / final_mask.size)*100
        self.sand_mask = final_mask
        return final_mask

    
    def pixel_vegetation(self, region, lambda_val=3):
        """Calculate the vegetation pixel value for a region"""
        # Create a mask for vegetation pixels such that the difference between the green and red channels is less than lambda_val
        GR_mask = np.abs(region[:,:,1] - region[:,:,2]) > lambda_val
        GB_mask = np.abs(region[:,:,1] >= region[:,:,0])
        mask = GR_mask & GB_mask
        # Count the number of vegetation pixels
        vegetation_pixels = np.sum(mask)
        # Print percentage of vegetation pixels
        print(f"Vegetation Pixels: {vegetation_pixels}")
        print(f"Percentage of Vegetation Pixels: {(vegetation_pixels / mask.size)*100:.2f}%")
        self.veg_mask = mask
        self.percent_veg = (vegetation_pixels / mask.size)*100
        return mask
    
    def pixel_water(self, region, lambda_val=3):
        """Calculate the water pixel value for a region"""
        # Create a mask for water pixels such that the blue channel is greater than the red channel
        BR_mask = region[:,:,0] - region[:,:,2] > lambda_val
        BG_mask = region[:,:,0] > region[:,:,1]
        mask = BR_mask & BG_mask
        # Count the number of water pixels
        water_pixels = np.sum(mask)
        # Print percentage of water pixels
        print(f"Water Pixels: {water_pixels}")
        print(f"Percentage of Water Pixels: {(water_pixels / mask.size)*100:.2f}%")
        self.percent_water = (water_pixels / mask.size)*100
        self.water_mask = mask
        return mask
    
    
    def plot_region_and_mask(self, region, mask, title="Sand Mask"):
        """
        Plots the original region (converted to RGB) and the masked region side by side.

        Parameters:
            region (numpy.ndarray): The original BGR region as a numpy array.
            mask (numpy.ndarray): The mask to overlay on the region.
            title (str): The title for the mask subplot.
        """
        # Convert BGR to RGB for display
        print(f"Region shape: {region.shape}")
        rgb_region = region[:, :, ::-1]

        # Create a masked image (e.g., overlay mask on the original image)
        masked_region = rgb_region.copy()
        masked_region[~mask] = 0  # Set non-masked areas to black

        # Plot the subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_region)
        axes[0].set_title("Original Region")
        axes[0].axis("off")

        axes[1].imshow(masked_region)
        axes[1].set_title(title)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        
        
    def plot_SVW_mask(self, region):
        """
        Plots the original region (converted to RGB) and the masked region side by side.

        Parameters:
            region (numpy.ndarray): The original BGR region as a numpy array.
            mask (numpy.ndarray): The mask to overlay on the region.
            title (str): The title for the mask subplot.
        """
        # Convert BGR to RGB for display
        print(f"Region shape: {region.shape}")
        rgb_region = region[:, :, ::-1]
        
        s_mask = self.sand_mask
        v_mask = self.veg_mask
        w_mask = self.water_mask

        # Create a masked image (e.g., overlay mask on the original image)
        s_masked_region = rgb_region.copy()
        s_masked_region[~s_mask] = 0  # Set non-masked areas to black
        v_masked_region = rgb_region.copy()
        v_masked_region[~v_mask] = 0  # Set non-masked areas to black
        w_masked_region = rgb_region.copy()
        w_masked_region[~w_mask] = 0  # Set non-masked areas to black

        # Plot the subplots
        fig, axes = plt.subplots(1, 4, figsize=(22, 10))
        axes[0].imshow(rgb_region)
        axes[0].set_title("Original Region")
        axes[0].axis("off")

        axes[1].imshow(s_masked_region)
        axes[1].set_title("Sand Mask")
        axes[1].axis("off")
        
        axes[2].imshow(v_masked_region)
        axes[2].set_title("Vegetation Mask")
        axes[2].axis("off")
        
        axes[3].imshow(w_masked_region)
        axes[3].set_title("Water Mask")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()


    
    def swv_vectorization(self):
        if self.segmented_dataframe is None:
            raise ValueError("Segmented dataframe not available. Please run segment_images() first.")
        else:
            df = self.segmented_dataframe.copy()
            
        final_df = pd.DataFrame()

        for idx, row in df.iterrows():
            region_array = row['Region_Array']
            self.pixel_sand(region_array, lambda_val=8)
            self.pixel_vegetation(region_array, lambda_val=8)
            self.pixel_water(region_array, lambda_val=10)
            svw = (self.percent_sand, self.percent_veg, self.percent_water)
            row['SVW'] = svw
            # concatenate the row to the final dataframe
            final_df = pd.concat([final_df, row], axis=1)
            
            
        return final_df.T
            

    def get_segmented_dataframe(self):
        """Returns the segmented dataframe, building it if necessary"""
        if self.segmented_dataframe is None:
            raise ValueError("Segmented dataframe not available. Please run segment_images() first.")
        return self.segmented_dataframe
    
        
    # get average blue, green, red values for each image
    # add to image dataframe
    def get_image_means(self):
        for img_date in self.img_dict.keys():
            img = self.img_dict[img_date]
            # print(f"Image array shape: {img.shape}")
            blue = np.mean(img[:,:,0])
            green = np.mean(img[:,:,1])
            red = np.mean(img[:,:,2])
            
            # use Datetime index for assignment
            if self.image_df is not None and img_date in self.image_df.index:
                # print(blue, green, red)
                self.image_df.loc[img_date, 'blue'] = np.float64(blue)
                self.image_df.loc[img_date, 'green'] = np.float64(green)
                self.image_df.loc[img_date, 'red'] = np.float64(red)
            else:
                print(f"Warning: image_df is not properly initialized or img_date {img_date} is not in the index.")
        return self.image_df
    
    
    ##################################################
    ##################################################
    # Plotting functions
    def plot_image_grid(self, img_date=None):
        # if no image data, use first image
        if img_date is None:
            img_date = list(self.ROI_dict.keys())[0]
        img = self.ROI_dict[img_date]
        grid_size = self.grid_size
        # plot 2by2 subplot of pure BGR images and the original converted to (RGB) format for matplotlib
        # overlay the grid on each image from the top-left corner
        # use the grid size to determine the number of grid lines without axvline and axhline
        fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
        # [0,0] pure blue (only color map the blue channel) do not color map the other channels
        axs[0, 0].imshow(img[:,:,0], cmap='Blues')
        # [0,1] pure green
        axs[0, 1].imshow(img[:,:,1], cmap='Greens')
        # [1,0] pure red
        axs[1, 0].imshow(img[:,:,2], cmap='Reds')
        # [1,1] original image
        axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # add grid lines to each image
        for ax in axs.flatten():
            ax.set_xticks(np.arange(0, img.shape[1], grid_size))
            ax.set_yticks(np.arange(0, img.shape[0], grid_size))
            ax.grid(which='both')
            
        # angle the x-axis labels for better readability on the shared axis
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.show()
        

    def plot_bgr_clusters(self):
        if self.segmented_dataframe is None:
            raise ValueError("Segmented dataframe not available. Please run segment_images() first.")
        
        # Extract BGR data
        avg_bgr_data = self.segmented_dataframe['Avg_BGR'].apply(pd.Series)  # Split tuples into columns
        avg_bgr_data.columns = ['Avg_B', 'Avg_G', 'Avg_R']
        median_bgr_data = self.segmented_dataframe['Median_BGR'].apply(pd.Series)
        median_bgr_data.columns = ['Median_B', 'Median_G', 'Median_R']
        
        # Add Region column for hover information
        avg_bgr_data['Region'] = self.segmented_dataframe['Region']
        median_bgr_data['Region'] = self.segmented_dataframe['Region']
        
        # Plot 1: Average BGR Scatter Plot
        fig_avg = go.Figure()
        fig_avg.add_trace(
            go.Scatter3d(
                x=avg_bgr_data['Avg_B'],
                y=avg_bgr_data['Avg_G'],
                z=avg_bgr_data['Avg_R'],
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.8),
                name='Average BGR',
                hovertemplate=(
                    "Region: %{customdata}<br>" +
                    "B: %{x}<br>G: %{y}<br>R: %{z}<extra></extra>"
                ),
                customdata=avg_bgr_data['Region']
            )
        )
        fig_avg.update_layout(
            scene=dict(
                xaxis_title='Blue',
                yaxis_title='Green',
                zaxis_title='Red'
            ),
            width=1000,  # Control plot width
            height=800,  # Control plot height
            title="3D Scatter Plot of Average BGR Values",
            legend_title="BGR Metrics"
        )
        
        # Plot 2: Median BGR Scatter Plot
        fig_median = go.Figure()
        fig_median.add_trace(
            go.Scatter3d(
                x=median_bgr_data['Median_B'],
                y=median_bgr_data['Median_G'],
                z=median_bgr_data['Median_R'],
                mode='markers',
                marker=dict(size=5, color='green', opacity=0.8),
                name='Median BGR',
                hovertemplate=(
                    "Region: %{customdata}<br>" +
                    "B: %{x}<br>G: %{y}<br>R: %{z}<extra></extra>"
                ),
                customdata=median_bgr_data['Region']
            )
        )
        fig_median.update_layout(
            scene=dict(
                xaxis_title='Blue',
                yaxis_title='Green',
                zaxis_title='Red'
            ),
            width=1000,  # Control plot width
            height=800,  # Control plot height
            title="3D Scatter Plot of Median BGR Values",
            legend_title="BGR Metrics"
        )
        
        # Show the plots
        fig_avg.show()
        fig_median.show()
        
        
        
        
    def img_display(self, img_date=None):
        # Display the image
        if img_date is None:
            img_date = list(self.img_dict.keys())[0]
        img = self.img_dict[img_date]
        # show numpy array as rgb image with cv2
        cv2.imshow("Image Test",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # station_name = 'oakisland_west'
    station_name = 'jennette_north'
    # station_name = 'currituck_hampton_inn'
    getImg = GetImages(station_name)
    print(getImg.get_image_dir()) # get the image directory
    getImg.get_image_dir() # get the image directory
    getImg.set_image_type('timex') # set the image type
    getImg.get_image_list()[:3] # get the image list
    getImg.parse_image_datetime()[:3] # parse the image datetime
    getImg.create_image_df() # create the image dataframe
    # print(f"Total images in {station_name} directory:\n{getImg.image_df.info()}")
    
    # import a subset of images
    dt_range = getImg.get_date_range('2024-07-14 00:00:00', '2024-07-15 00:00:00') # get the date range
    # dt_range = getImg.get_date_range() # get the date range
    getImg.get_image_range(dt_range) # get the image range
    # print(f"Images in Date Range:\n{getImg.image_range}")
    
    # get images for analysis
    # Note that images are imported with opencv2 and are Blue-Green-Red (BGR) format
    # Plotting in matplotlib requires converting to Red-Green-Blue (RGB) format (cv2.COLOR_BGR2RGB)
    imgs = getImg.get_images() # get the images
    
    IA = ImageAnalysis(imgs, getImg.image_range.copy()) # image analysis
    IA.get_image_means() # get the image
    print(IA.img_dict.keys()) # image dictionary
    img_dates = list(IA.img_dict.keys()) # image dates
    print(f"First image in dictionary:\n{type(IA.img_dict[img_dates[1]])}") # first image in dictionary
    print(f"Image Dataframe:\n{IA.image_df.head()}") # image dataframe
    
    # IA.set_ROI(100, 100, 200, 200) # set region of interest
    grid_size = 400
    img_segs = IA.segment_images(grid_size=grid_size) # segment the images
    print(f"With image size {IA.img_dict[img_dates[1]].shape} and gridsize {grid_size}, there are {len(img_segs['2024-07-14 10:05:21'].keys())} segments.") # segmented image data
    # print(f"Segmented image data:\n{img_segs['2024-07-14 10:05:21']['region_0_0']}") # segmented image data
    
    # IA.build_segmented_dataframe()
    IA.add_bgr_stats() # add BGR stats
    print(f"Segmented DataFrame:\n{IA.segmented_dataframe.iloc[35:85,:]}") # segmented dataframe
    
    # IA.img_display(img_dates[1]) # display the image
    