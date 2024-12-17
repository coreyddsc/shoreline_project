import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
# Additionally, build directory tools to handle the JSON files, configs, and images name/dates.
# Build date labelers for arrays and images on matplotlib subplots.

class ShorelineDataProcessor:
    def __init__(self, station_name):
        self.station_name = station_name
        self.config_directory = self._get_config_directory()
        self.data_directory = self._get_data_directory()
        self.transects_json_df = pd.DataFrame()  # Ensure this is initialized
        self.coords_df = pd.DataFrame()
        self.station_info_df = pd.DataFrame()  # DataFrame to store station info
        self.transects_df = pd.DataFrame()  # DataFrame to store transect data
        self.orientation = None # Initialize orientation of the ocean to the camera
        self.vidSnap = None # Initialize the video snapshot
        
        self.get_shoreline_jsons()
        self.get_station_info()

    def _get_config_directory(self):
        # Get the directory for the config.json files
        cwd = os.getcwd()
        return os.path.join(cwd, 'configs')

    def _get_data_directory(self):
        # Go up one directory and then into transect_jsons/{station_name}
        cwd = os.getcwd()
        return os.path.join(cwd, 'transect_jsons', self.station_name)

    def get_station_info(self):
        # Construct the path to the config.json file
        station_path = os.path.join(self.config_directory, f'{self.station_name}.config.json')

        # Handle alternative path if the stationPath doesn't exist
        if not os.path.exists(station_path):
            alt_path = os.path.join(f'.\configs\{self.station_name}.config.json')
            if os.path.exists(alt_path):
                station_path = alt_path
            else:
                raise FileNotFoundError(f"Configuration file for station '{self.station_name}' not found.")

        # Load the entire station configuration JSON into a DataFrame
        with open(station_path, 'r') as setup_file:
            station_info = json.load(setup_file)
        
        # Convert the station info dictionary to a DataFrame
        self.station_info_df = pd.DataFrame([station_info])
        return self.station_info_df
    
    # Station Orientation
    def get_station_orientation(self):
        # Ensure that station info is loaded
        if self.station_info_df.empty:
            raise ValueError("Station info is not loaded. Please run get_station_info() first.")
        
        # Extract the orientation value and store it as an attribute
        self.orientation = self.station_info_df.at[0, 'Orientation']
        return self.orientation

    # Station Transects
    def get_station_transects(self):
        # Ensure that station info is loaded
        if self.station_info_df.empty:
            raise ValueError("Station info is not loaded. Please run get_station_info() first.")

        # Extract and convert transects to NumPy arrays
        dune_line_interp = np.asarray(self.station_info_df.at[0, 'Dune Line Info']['Dune Line Interpolation'])
        transects_x = np.asarray(self.station_info_df.at[0, 'Shoreline Transects']['x'])
        transects_y = np.asarray(self.station_info_df.at[0, 'Shoreline Transects']['y'])

        # Combine the transects into a DataFrame with each column containing matching values
        self.transects_df = pd.DataFrame({
            'Transects X': transects_x.flatten(),
            'Transects Y': transects_y.flatten()
        })

        # Add dune line interpolation to the DataFrame if its length matches X and Y
        if len(dune_line_interp) == len(transects_x):
            self.transects_df['Dune Line Interpolation'] = dune_line_interp.flatten()

        return self.transects_df

#######################################################################################################
# These functions rely of the detected shoreline jsons. 

    def get_shoreline_jsons(self):
        # Initialize an empty list to store the data
        data = []

        # Iterate through all files in the data directory
        print(self.data_directory)
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_directory, filename)
                
                # Open and read the JSON file
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                
                # Append the entire JSON data as a dictionary
                json_data["Filename"] = filename  # Keep track of the filename
                data.append(json_data)

        # Convert the list of data to a DataFrame and store it as self.df
        self.transects_json_df = pd.DataFrame(data)
        return self.transects_json_df
    
    #     return self.coords_df
    def get_shoreline_coords(self):
        # Ensure that the main DataFrame is not empty
        if self.transects_json_df.empty:
            raise ValueError("DataFrame is empty. Please run get_shoreline_jsons() first.")
        
        # Ensure that orientation is set
        if self.orientation is None:
            self.get_station_orientation()

        # Explode the Shoreline Points to separate rows for each point
        exploded_df = self.transects_json_df.explode("Shoreline Points")

        # Extract x, y coordinates from the Shoreline Points column
        exploded_df[['X', 'Y']] = pd.DataFrame(exploded_df['Shoreline Points'].tolist(), index=exploded_df.index)

        # Convert the 'Time Info' to datetime
        exploded_df['Datetime'] = pd.to_datetime(exploded_df['Time Info'])
        print(f'Shape of exploded_df: {exploded_df.shape}')
        print(f'Info: {exploded_df.loc[:, ["Time Info", "X", "Y"]]}')
        # # Handle orientation-specific logic
        if self.orientation == 0:
            # Drop duplicate (Datetime, X) pairs, keeping only the first observation
            exploded_df = exploded_df.drop_duplicates(subset=['Datetime', 'X'], keep='first')

            self.coords_df = exploded_df.pivot(index='Datetime', columns='X', values='Y')
        else:
            # Drop duplicate (Datetime, X) pairs, keeping only the first observation
            exploded_df = exploded_df.drop_duplicates(subset=['Datetime', 'Y'], keep='first')
            # Pivot the DataFrame so that y-coordinates become columns
            self.coords_df = exploded_df.pivot(index='Datetime', columns='Y', values='X')
            

        print(f'Shape of coords_df: {self.coords_df.shape}')

        return self.coords_df

    






