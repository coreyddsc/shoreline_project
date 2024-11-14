import os
import cv2
import math
import numpy as np
import requests
from io import BytesIO
import time
from getTimexShoreline import *
from read_jsons import *
from webcoos_request import *
import matplotlib.pyplot as plt
from matplotlib import cm


from skimage import io
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu, threshold_local
from skimage.filters.rank import otsu
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.draw import polygon

class ImageProcessor:
    ##########################################################
    def __init__(self, stationName='oakisland_west'):
        # self.vidName = vidName
        self.stationName = stationName
        self.base_dir = os.path.join(os.getcwd(), 'images')
        self.station_dir = os.path.join(self.base_dir, self.stationName)
        self.timex_dir = os.path.join(self.station_dir, 'timex')
        self.brt_dir = os.path.join(self.station_dir, 'bright')
        self.avg_dir = os.path.join(self.station_dir, 'average')
        # self.vidPath = os.path.join(os.getcwd(), self.vidName)
        self.subSampleRate = 10
        self.tgtFrameCount = 0
        self.fps = 0
        self.totFrameCount = 0
        self.vidLength = 0
        self.w = 0

        self.photoAvg = None  # To store the time-averaged image
        self.photoBrt = None  # To store the brightest pixel image
        self.timexName = None  # To store the filename for the time-averaged image
        
        self.tranSL = None  # To store the shoreline transects
        self.fig_tranSL = None  # To store the figure of the transects

        self.current_time_info = None  # Initialize current_time_info
        
        self.timexImgs = []  # Initialize list to store timex images
        self.timexLabels = [] # Initialize list to store timex labels
        self.timexRMBs = []  # Initialize list to store red-minus-blue images
        self.timexOtsus = []  # Initialize list to store Otsu threshold images
        self.timexLocalOtsus = []  # Initialize list to store local Otsu threshold images
        self.timexHighpasses = []  # Initialize list to store highpass filtered images
        self.timexEdges = []  # Initialize list to store edge transformed images
        
        self.contours = None  # Initialize contours attribute

        self._initialize_directories()
 

    ##################################
    def _initialize_directories(self):
        """Create the necessary directories for storing images."""
        for dir_path in [self.base_dir, self.station_dir, self.timex_dir, self.brt_dir, self.avg_dir]:
            print(f"Attempting to create directory: {dir_path}")
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    print(f"Created directory: {dir_path}")
                except Exception as e:
                    print(f"Failed to create directory {dir_path}: {e}")
            else:
                print(f"Directory already exists: {dir_path}")


    #########################
    def download_video(self, vidName = None):
        if vidName is None:
            print("No video URL provided. Please provide a valid video URL.")
        elif isinstance(vidName, str):
            self.vidName = vidName
            self.vidPath = os.path.join(os.getcwd(), self.vidName)
        #continue with the download process
        """Downloads the video from the URL and saves it to a temporary file."""
        if self.vidName.startswith('http://') or self.vidName.startswith('https://'):
            start_time = time.time()  # Start the timer
            response = requests.get(self.vidName, stream=True)
            if response.status_code == 200:
                video_data = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    video_data.write(chunk)
                video_data.seek(0)  # Reset the file pointer to the beginning
                video_name = self.vidName.split('/')[-1].replace('.mp4', '')
                self.vidPath = f'{video_name}.mp4'
                with open(self.vidPath, 'wb') as f:
                    f.write(video_data.getbuffer())
                download_duration = time.time() - start_time
                print(f"Video download complete. Duration: {download_duration:.2f} seconds.")
            else:
                raise ValueError(f"Failed to download video: {response.status_code}")
        else:
            self.vidPath = os.path.join(os.getcwd(), self.vidName)
            if not os.path.isfile(self.vidPath):
                raise FileNotFoundError(f"Video file not found: {self.vidPath}")

        self._load_video()


    #######################
    def _load_video(self):
        """Load the video and extract relevant metadata."""
        vidObj = cv2.VideoCapture(self.vidPath)
        self.fps = vidObj.get(cv2.CAP_PROP_FPS)
        self.totFrameCount = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vidLength = math.floor((self.totFrameCount / self.fps) - 1)
        self.tgtFrameCount = math.floor(self.vidLength / self.subSampleRate)
        self.w = 1 / self.tgtFrameCount
        vidObj.release()


    ###############################
    def generate_timex_image(self):
        """Generate the time-averaged image product."""
        vidObj = cv2.VideoCapture(self.vidPath)
        fname, _ = os.path.splitext(os.path.basename(self.vidPath))
        self.timexName = os.path.join(self.timex_dir, fname + '-timex.jpeg')

        end = self.tgtFrameCount - 1

        success, snap = vidObj.read()
        if not success:
            raise ValueError(f"Failed to read the first frame from video: {self.vidPath}")
        
        self.photoAvg = np.zeros(np.shape(snap))

        for i in range(end):
            timeSet = i * 1000 * self.subSampleRate
            vidObj.set(cv2.CAP_PROP_POS_MSEC, timeSet)
            success, snap = vidObj.read()

            if not success:
                continue

            snapd = np.array(snap).astype(float)
            if i == 0:
                self.photoAvg = self.w * snapd
            else:
                self.photoAvg += self.w * snapd

        # Scale/save time-averaged image product
        avgScale = self.photoAvg.max()
        self.photoAvg = np.uint8(254 * (self.photoAvg / avgScale))
        cv2.imwrite(self.timexName, self.photoAvg)

        # Convert to RGB
        self.photoAvg = cv2.cvtColor(self.photoAvg, cv2.COLOR_BGR2RGB)
        vidObj.release()


    #########################################
    def generate_brightest_pixel_image(self):
        """Generate the brightest pixel image product."""
        vidObj = cv2.VideoCapture(self.vidPath)

        end = self.tgtFrameCount - 1

        success, snap = vidObj.read()
        if not success:
            raise ValueError(f"Failed to read the first frame from video: {self.vidPath}")
        
        self.photoBrt = np.zeros(np.shape(snap))

        for i in range(end):
            timeSet = i * 1000 * self.subSampleRate
            vidObj.set(cv2.CAP_PROP_POS_MSEC, timeSet)
            success, snap = vidObj.read()

            if not success:
                continue

            snapd = np.array(snap).astype(float)
            if i == 0:
                self.photoBrt = snapd
            else:
                self.photoBrt = np.where(snapd > self.photoBrt, snapd, self.photoBrt)

        # Scale/save brightest pixel image product
        brtScale = self.photoBrt.max()
        self.photoBrt = np.uint8(254 * (self.photoBrt / brtScale))
        brtName = os.path.join(self.brt_dir, os.path.basename(self.timexName).replace('-timex', '-brt'))
        cv2.imwrite(brtName, self.photoBrt)

        # Convert to RGB
        self.photoBrt = cv2.cvtColor(self.photoBrt, cv2.COLOR_BGR2RGB)
        vidObj.release()
        
        
    ######################################        
    def generate_timexavg_shoreline(self):   
        """
        Generate the time-averaged image, then extract and return the shoreline.
        The getTimexShoreline.py methods have not been refactored.
        """
        # Ensure that timex image is generated
        if self.timexName is None:
            self.generate_timex_image()

        # Process the timex image to extract the shoreline using the old getTimexShoreline method
        self.tranSL, self.fig_tranSL = getTimexShoreline(self.stationName, self.timexName)

        return self.tranSL, self.fig_tranSL
        
        
    ###################    
    def clean_up(self):
        """Deletes the temporary video file after processing."""
        if self.vidPath and os.path.exists(self.vidPath) and self.vidName.startswith('http'):
            os.remove(self.vidPath)
            print(f"Deleted temporary video file: {self.vidPath}")
            
    #######################
    def create_img_products(self, url):
        """
        Create image products from the video URL.
        """
        self.download_video(url)
        self.generate_timex_image()
        self.generate_brightest_pixel_image()
        self.generate_timexavg_shoreline()
        self.clean_up()        
            
            
      
     
    ###########################  
    def get_timex_image(self):
        """
        Grabs processed timex image from folder for given station.
        Returns a dictionary of timex numpy rgb numpy arrays with keys as the image filenames.
        """
        cwd = os.getcwd()
        timex_dir = os.path.join(cwd, 'images', self.stationName, 'timex')
        timex_files = os.listdir(timex_dir)
        for img in timex_files:
            if img.endswith('.jpeg'):
                self.timexImg = io.imread(os.path.join(timex_dir, img))
                self.timexImgs.append(self.timexImg)
                
                self.timexLabel = img
                self.timexLabels.append(self.timexLabel)
                # self.timex_imgs[img] = self.timexImg
        if self.timexLabel is None:
            raise FileNotFoundError("No timex image found in the timex directory.")

        return self.timexImgs, self.timexLabels
    
    def rmb_transform(self, normalize=True):
        """
        rmb_transform: Red Minus Blue Transform
        
        This method will take the timex image and transform it into a red minus blue image. Must be a numpy rgb array.
        """
        for img in self.timexImgs:
            rmb = img[:, :, 0] - img[:, :, 2]
            
            if normalize:
                rmb = (rmb - rmb.min()) / (rmb.max() - rmb.min())
                # # Normalize the result to the range 0-255 for display
                # rmb = np.clip(rmb, 0, 255).astype(np.uint8)
                # rmb = cv2.normalize(rmb, None, 0, 255, cv2.NORM_MINMAX)
                self.timexRMBs.append(rmb)
            
            self.timexRMBs.append(rmb)
                
        return self.timexRMBs
    
    def highpass_filter(self):
        for img in self.timexRMBs:
            gaussian_blur = gaussian(img, sigma=10)
            highpass = img - gaussian_blur
            
            # Normalize the result to the range 0-255 for display
            # highpass = np.clip(highpass, 0, 255).astype(np.uint8)
            highpass = (highpass - highpass.min()) / (highpass.max() - highpass.min())
            self.timexHighpasses.append(highpass)
            
        return self.timexHighpasses
            
            
    def otsu_threshold(self):
        for img in self.timexRMBs:
            # Convert image to 8-bit unsigned integer (if not)
            img_ubyte = img_as_ubyte(img)
            otsu_thresh = threshold_otsu(img_ubyte)
            binary_otsu = img_ubyte >= otsu_thresh
            
            self.timexOtsus.append(binary_otsu)
        return self.timexOtsus
    
    def local_otsu_transform(self, radius=30):
        for img in self.timexRMBs:
            # convert image to 8-bit unsigned integer (if not)
            img_ubyte = img_as_ubyte(img)
            local_otsu = otsu(img_ubyte, disk(radius))
            
            # Apply local Otsu thresholding
            binary_local_ot = img_ubyte >= local_otsu
            
            self.timexLocalOtsus.append(binary_local_ot)
        return self.timexLocalOtsus
    
    def edge_transform(self):
        """
        Apply edge detection to the timex images.
        Apply the difference of the pixel values to the left of each pixel.
        """
        for img in self.timexImgs:
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection by subtracting each pixel value from the value of the pixel to the left
            convolved = np.zeros_like(gray)
            convolved[:, 1:] = gray[:, 1:] - gray[:, :-1]
            
            self.timexEdges.append(convolved)
            
        return self.timexEdges
                
    def plot_images(self, cmap='viridis'):
        """
        Plot the timex images and red-minus-blue images side by side.
        """
        if not self.timexImgs or not self.timexRMBs:
            raise ValueError("No images to display. Ensure that timexImgs and timexRMBs are populated.")
        
        num_images = len(self.timexImgs)
        fig, ax = plt.subplots(num_images, 6, figsize=(24, 6 * num_images))
        
        for i in range(num_images):
            timex = self.timexImgs[i]
            rmb = self.timexRMBs[i]
            otsu = self.timexOtsus[i]
            local_otsu = self.timexLocalOtsus[i]
            highpass = self.timexHighpasses[i]
            edges = self.timexEdges[i]
            
            # Display timex image
            ax[i, 0].imshow(timex)
            # ax[i, 0].set_title(f"Timex Image: {self.timexLabels[i]}", fontsize=9)
            # ax[i, 0].axis('off')
            
            # Display red-minus-blue image
            # Ensure rmb is a 2D array (grayscale) for correct colormap application
            if len(rmb.shape) == 3:
                rmb = rmb[:, :, 0]  # Use only one channel if rmb is 3D
            
            ax[i, 1].imshow(rmb, cmap=cmap, vmin=0, vmax=1)
            # ax[i, 1].set_title(f"Red Minus Blue Image: {self.timexLabels[i]}", fontsize=9)
            # ax[i, 1].axis('off')
            
            # Display Highpass filtered image
            ax[i, 2].imshow(highpass, cmap='gray')
            # ax[i, 2].set_title(f"Highpass Filter: {self.timexLabels[i]}", fontsize=9)
            # ax[i, 2].axis('off')
            
            # Display Otsu thresholded image
            ax[i, 3].imshow(otsu, cmap=cmap)
            
            # Display local Otsu thresholded image
            ax[i, 4].imshow(local_otsu, cmap=cmap)
            # ax[i, 4].set_title(f"Local Otsu: {self.timexLabels[i]}", fontsize=9)
            # ax[i, 4].axis('off')
            
            # Display edge transformed image
            ax[i, 5].imshow(edges, cmap='gray')
        
        plt.tight_layout()
        plt.show()           
  
if __name__ == "__main__":
    start_time = '2024-07-22 00:00:00'
    end_time = '2024-07-23 00:00:00'
    station_names = ['oakisland_west', 'jennette_north','currituck_hampton_inn']
    inv = get_inventory(start_time=start_time, end_time=end_time,station=station_names[0])
    url = inv[0]['data']['properties']['url']
    IP = ImageProcessor(stationName=station_names[0])
    IP.create_img_products(url)



