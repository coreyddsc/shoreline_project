import json
import requests
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

class GetShorelineJSONS:
    def __init__(self, station_name):
        self.station_name = station_name
        self.date = '/2024/08/08/'
        self.station_url = f'https://stage-ams.srv.axds.co/archive/jsonl/noaa/{station_name}/{self.date}'
        
    def set_date(self, date):
        # parse the date to the correct format
        self.date = date
        self.station_url = f'https://stage-ams.srv.axds.co/archive/jsonl/noaa/{self.station_name}/{self.date}'
        
    def build_date_range(self, start_date, end_date):
        # construct list with days between start and end date
        # dates should have format 'YYYY/MM/DD/'
        self.date_range = []
        for date in pd.date_range(start_date, end_date):
            self.date_range.append(date.strftime('%Y/%m/%d/'))
        
    def get_response(self):
        self.response = requests.get(self.station_url)

    def get_content(self):
        self.content = self.response.content

    def get_json_refs(self):
        # if has attribute error: object has no attribute 'response', run get_response() first
        if not hasattr(self, 'response'):
            self.get_response()
        self.json_refs = re.findall(r'href="([^"]*\.shoreline_otsu\.v1\.timex\.[^"]*)"', self.response.text)
        
    def get_json(self, json_ref):
        # print(self.station_url + json_ref)
        self.json_response = requests.get(self.station_url + json_ref)
        # print response status
        # print(self.json_response.status_code)
        self.json_content = self.json_response.json()
        
    def get_shoreline_orientation(self):
        # self.shoreline_orientation = self.json_content['detected_shoreline']['Orientation']
        if self.station_name == 'currituck_hampton_inn':
            self.shoreline_orientation = 0
        
    def get_time_info(self):
        self.time_info = self.json_content['detected_shoreline']['Time Info']
        # split on the period and add the Z to the end
        self.time_info = self.time_info.split('.')[0] + 'Z'
        
    def get_shoreline_points(self):
        self.shoreline_points = self.json_content['detected_shoreline']['Shoreline Points']
        self.xc = []
        self.yc = []
        for x, y in self.shoreline_points:
            self.xc.append(x)
            self.yc.append(y)
        if self.shoreline_orientation == 0:
            self.shoreline_points = pd.DataFrame([self.yc], columns=self.xc)
            
        self.shoreline_points.index = [self.time_info]
        
     
    # build final dataframe with all the shorelines   
    def get_shorelines(self):
        self.get_json_refs()
        self.shorelines_df = pd.DataFrame()
        for json_ref in self.json_refs:
            self.get_json(json_ref)
            self.get_shoreline_orientation()
            self.get_time_info()
            self.get_shoreline_points()
            self.shorelines_df = pd.concat([self.shorelines_df, self.shoreline_points])
            # print(self.shoreline_points)
            
    # build dataframe that using the date range to get all the shorelines
    def get_shorelines_date_range(self, verbose=False):
        self.shorelines_df = pd.DataFrame()
        for date in self.date_range:
            self.set_date(date)
            if verbose:
                print(f'Getting Shorelines for Date: {self.date}')
                print(f'Current Station URL: {self.station_url}')
            self.get_response()
            self.get_json_refs()
            
            for json_ref in self.json_refs:
                self.get_json(json_ref)
                self.get_shoreline_orientation()
                self.get_time_info()
                self.get_shoreline_points()
                self.shorelines_df = pd.concat([self.shorelines_df, self.shoreline_points])

    # check if the shoreline is smooth
    # Function to compute the second derivative
    def compute_second_derivative_row(self, x_values, y_values):
        # Compute first derivative using finite differences
        first_derivative = np.diff(x_values) / np.diff(y_values)
        
        # Compute second derivative using finite differences of the first derivative
        second_derivative = np.diff(first_derivative) / np.diff(y_values[:-1])
        
        return second_derivative


    # Function to compute smoothness metric for each row
    def compute_smoothness_for_slx(self):
        self.slx = self.shorelines_df.copy()
        smoothness_scores = []
        
        for index, row in self.slx.iterrows():
            y_values = row.index.values.astype(float)  # Columns are the Y values
            x_values = row.values  # Row values are the X values

            
            second_derivative = self.compute_second_derivative_row(x_values, y_values)
            
            # Compute smoothness score based on the norm of the second derivative
            smoothness_score = np.linalg.norm(second_derivative)
            smoothness_scores.append(smoothness_score)
        
        # Add smoothness scores as a new column in the dataframe
        self.slx['smoothness_score'] = smoothness_scores

        # Normalize the smoothness scores
        self.slx['smoothness_score'] = (self.slx['smoothness_score'] - self.slx['smoothness_score'].min()) / (self.slx['smoothness_score'].max() - self.slx['smoothness_score'].min())
        
        return self.slx
    
    def qaqc_smoothness_check(self):
        # check if the shoreline is smooth
        pass
            
    # Plot Smoothness Quantiles
    # Function to plot the curve with smoothness score for each quantile
    def plot_smoothness_by_quantile(self, quantiles=np.linspace(0, 1, 20)):
        slx_with_smoothness = self.compute_smoothness_for_slx()
        # Sort the dataframe by smoothness score
        sorted_slx = slx_with_smoothness.sort_values(by='smoothness_score').reset_index(drop=True)
        
        # Calculate the quantile values for smoothness scores
        quantile_values = sorted_slx['smoothness_score'].quantile(quantiles)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16), sharey=True)
        axes = axes.flatten()  # Flatten to iterate over the 10 axes
        
        for i, quantile in enumerate(quantiles):
            # Find the index of the first curve that matches the quantile
            quantile_index = sorted_slx[sorted_slx['smoothness_score'] >= quantile_values[quantile]].index[0]
            
            # Get the curve (X, Y) for the selected quantile
            row = sorted_slx.iloc[quantile_index]
            x_values = row.values[:-1]  # X-values (row values)
            y_values = row.index[:-1].astype(float)[::-1]  # Y-values (column index), reversed
            
            smoothness_score = row['smoothness_score']

            if self.shoreline_orientation == 0:
                x_values, y_values = y_values, x_values
            
            # Plot the curve
            axes[i].plot(x_values, y_values, label=f'Smoothness: {smoothness_score:.2f}')
            axes[i].set_title(f'Quantile: {quantile * 100:.0f}%')
            axes[i].legend()
            # axes[i].set_xlabel('X (Row Values)')
            
            # set x-axis limit [0,450]
            axes[i].set_xlim(0,450)  
    
        # Set figure title
        fig.suptitle('Shoreline Smoothness Score by Quantile')
        # Set common Y label for all plots, offset left
        fig.supylabel('Image Shoreline Extraction Transections (Y)')
        # Set common X label for all plots, offset down
        fig.supxlabel('Image Shoreline Extraction Transections (X)')

        plt.tight_layout()
        plt.show()
        
        
if __name__ == '__main__':
    station_name = 'currituck_hampton_inn'
    get_shoreline_jsons = GetShorelineJSONS(station_name)
    get_shoreline_jsons.set_date('2024/08/10/')
    print(get_shoreline_jsons.station_url)
    get_shoreline_jsons.get_json_refs()
    print(get_shoreline_jsons.json_refs)
    print(get_shoreline_jsons.station_url + get_shoreline_jsons.json_refs[3])
    get_shoreline_jsons.get_json(get_shoreline_jsons.json_refs[3])
    get_shoreline_jsons.get_shoreline_orientation()
    print(get_shoreline_jsons.shoreline_orientation)
    get_shoreline_jsons.get_time_info()
    print(get_shoreline_jsons.time_info)
    get_shoreline_jsons.get_shorelines()
    print(get_shoreline_jsons.shorelines_df)
    