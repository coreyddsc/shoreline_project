import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from PIL import Image

class ShorelineTrend:
    def __init__(self, data):
        self.data = data.copy()
        self.range_mapping = None
        self.domain_mapping = None
        self.domain_param = None
        self.domain_subsets = None
        self.segment_dates = None # can be used to store a list or dictionary of dates to be used a keys and labels
        self.domain_segments = None
        
        self.model_data = None # can be used to store the data used for the current model
        self.model_results = {} # can be used to store the results of the current model
        self.domain_model_results = [] # can be used to store the results of the domain model

    # Parameterize the domain and set the domain_index_keys
    def parameterize_domain(self, report=False):
        """
        - DataFrame cannot have missing values in rows
        - DataFrame must have a datetime index in format YYYY-MM-DD HH:MM:SS
        - Start date must be in the format YYYY-MM-DD
        - End date must be in the format YYYY-MM-DD, but is not required.

        This function will return a DataFrame subset of the original DataFrame that is between the start and end dates.
        A new column called ts_domain is initialized with the first record being 0 and each subsequent record being the time in hours from the start date to the record date.
        """

        # Copy the DataFrame to avoid modifying the original
        df = self.data.copy()

        # Check if the DataFrame has missing values, drop them if they exist
        if df.isnull().values.any():
            df = df.dropna()
            if report:
                print("Missing values were found and dropped.")

        # Create ts_domain column using total hours
        df.loc[:, 'Time Domain'] = (df.index - df.index[0]).total_seconds() / 3600
        
        self.domain_param = df.loc[:, ['Time Domain']]
        return self.domain_param
    
    # creates a transect-index mapping for the range to be used in reference with
    # numpy array dimensions.
    def get_range_mapping(self):
        df = self.data.copy()
        cols = df.columns
        self.range_mapping = dict(zip(cols, range(len(cols))))
        return self.range_mapping
        
    # creates a datetime-value mapping for the domain
    def get_domain_mapping(self):
        df = self.data.copy()
        df.loc[:, 'Time Domain'] = (df.index - df.index[0]).total_seconds() / 3600
        # zip index and domain values into a dictionary for self.domain_index_keys
        self.domain_mapping = dict(zip(df.index, df['Time Domain']))
        return self.domain_mapping
    

    # get domain values from datetime index keys in a dataframe
    def get_domain_values(self, df = None):
        """
        Get the domain values from the datetime index keys in a DataFrame.
        """
        return df.index.map(self.domain_mapping)
    
    # get key date nearest to input value from domain_mapping dictionary
    def get_nearest_date(self, value):
        key = min(self.domain_mapping, key=lambda x: abs(self.domain_mapping[x] - value))
        print(f"Nearest Date to Value {value} is {key} with Domain Value {self.domain_mapping[key]}")
        return key
    
    
    # returns the subset of the domain from the start date to the end date
    def subset_domain(self, start_date=None, end_date=None):
        """
        Subset the data based on the start and end dates.
        """
        
        df = self.data.copy()
        
        if start_date is not None:
            if end_date is not None:
                subset = df.loc[start_date:end_date]
            else:
                subset = df.loc[start_date:]
        else:
            if end_date is not None:
                subset = df.loc[:end_date]
            else:
                subset = df
        self.domain_subsets = subset
        return self.domain_subsets
    
    # stores the dates for segmentation. Can store dictionary with keys and labels for event segmentation.
    def segmentation_dates_mapping(self, dates):
        """Stores the dates for segmentation."""
        self.segment_dates = dates
        return self.segment_dates

    # segments the domain based on the dates
    def segment_domain(self, dates = None, subset=False, trim=False):
        """
        Segment the data based on a list of dates.
        """
        segments = []
        
        if subset:
            df = self.domain_subsets.copy()
        else:
            df = self.data.copy()

        if dates == None:
            if self.segment_dates != None:
                dates = self.segment_dates
            else:
                raise ValueError("No dates were provided for segmentation.")

        if type(dates) == dict:
            # get the keys from the dictionary
            dates = list(dates.keys())
        
        if trim:
            for i in range(len(dates) - 1):
                start_date = dates[i]
                end_date = dates[i + 1]
                segment = df.loc[start_date:end_date]
                segments.append(segment)
        else:
            for i in range(len(dates)+1):
                print(i)    
                if i == 0:
                    segment = df.loc[df.index[0]:dates[i]]
                elif i > 0 and i < len(dates):
                    segment = df.loc[dates[i-1]:dates[i]]
                elif i == len(dates):
                    segment = df.loc[dates[-1]:df.index[-1]]
                    
                segments.append(segment)
                
                
        self.domain_segments = segments
        return self.domain_segments
    
        
    def fit_linear_model(self, transects = None, report=False):
        # model data is whatever data is being used for the model
        # data should be processed before fitting the model
        
        if self.model_data is not None:
            df = self.model_data.copy()
        else:
            df = self.data.copy()
            
        # use datetime index in df to get the correspond time domain values from the domain_mapping dictionary
        X = df.index.map(self.domain_mapping)     
            
        _, col = df.shape
        
        if transects == None:
            idy = int(round(col/2)) # default to middle transect
            rm_items = list(self.range_mapping.items())
            transects = [int(rm_items[idy][0])]
            y = df.iloc[:, [idy]] # default to middle transect 
        elif type(transects) == int:
            # get index for transect value from range_mapping dictionary
            if transects not in self.range_mapping:
                raise ValueError(f"Transect {transects} is not in the range mapping.")
            else:
                idy = self.range_mapping[transects]
                transects = [transects]
                y = df.iloc[:, [idy]]
        elif type(transects) == list:
            idy = [self.range_mapping[t] for t in transects]
            y = df.iloc[:, idy]
        elif transects == 'all':
            transects = list(self.range_mapping.keys())
            transects = [int(t) for t in transects]
            y = df.copy()
            
        # convert X and Y to numpy arrays
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        _, cols = y.shape

        if type(transects) == int:
            transects = [transects]
            
        # to keep the results space clean, reset self.model_results
        # alternate storage methods will be added in the future via other functions
        self.model_results = {}
        
        current_model = {}
        
        for col in range(cols):
            y_col = y[:, col]
            model = sm.OLS(y_col, sm.add_constant(X)).fit()
            transect = transects[col]
            current_model[f'model_t{transect}'] = model
            self.model_results[f'model_t{transect}'] = model
    
              
        if report:
            print(f"Model Data: {df.shape}")
            print(f"X has datatype {type(X)} and Y has datatype {type(y)}")
            print(f"Shape of Y data for transect {transects}:\n{y.shape}")
            print(f"Shape of X data: {X.shape} with values:\n{X[:5]}")
            
        return current_model
    
    
    def fit_domain_segments(self, transects = None, report=False):
        # if we want to control the domain segments we can use the segment_domain() method and add parameters to the fit_domain_segments() method
        # domain_segments = self.segment_domain()
        
        # to keep the results space clean, reset self.domain_model_results
        # alternate storage methods will be added in the future via other functions
        self.domain_model_results = []
        
        for segment in self.domain_segments:
            self.model_data = segment
            segment_model = self.fit_linear_model(transects=transects, report=report)
            
            # Store the segment model results using the specified keys
            segment_start = segment.index[0]
            segment_end = segment.index[-1]
            
            segment_results = {
                "start": segment_start,
                "end": segment_end,
                "model": segment_model  # Store the model as it is
            }
            
            self.domain_model_results.append(segment_results)
        
        return self.domain_model_results
    

    # JSON results can be used to create a report or dashboard for the user to view the results of the analysis. As well as plots and metrics for each model.

    
    def get_model_results(self):
        return self.model_results
    
    def get_domain_model_results(self):
        return self.domain_model_results
    
    
    # build out methods to get results and store them in a json format for easy access.
        

    def plot_segmented_domain_model(self, report=False, ax = None):
        dmr = self.domain_model_results.copy()
        
        if ax is None:
            # Create a figure and axis for plotting
            fig, ax = plt.subplots(figsize=(18, 6))
        
        # Define colors for each unique model key
        unique_keys = set()
        for segment in dmr:
            unique_keys.update(segment['model'].keys())
        colors = plt.cm.get_cmap('tab10', len(unique_keys))
        color_map = {key: colors(i) for i, key in enumerate(unique_keys)}
        
        # Define markers for each unique model key
        markers = ['o', 'x', '+', 's', 'D', '^', 'v', '<', '>', 'p', '*']
        marker_map = {key: markers[i % len(markers)] for i, key in enumerate(unique_keys)}
        
        # Handle the case where self.segment_dates is either a dictionary or a list
        if isinstance(self.segment_dates, dict):
            unique_dates = list(self.segment_dates.keys())
            event_labels = list(self.segment_dates.values())  # Get the labels for events
        else:
            unique_dates = self.segment_dates
            event_labels = None  # No labels if it's not a dictionary

        # Convert unique dates to matplotlib date numbers
        date_nums = sorted(set(mdates.date2num(pd.to_datetime(date)) for date in unique_dates))  # Ensure sorted and unique dates
        print(f"Segmentation dates (in matplotlib date format): {date_nums}")

        # Add vertical lines for the segmentation dates
        for i, date_num in enumerate(date_nums):
            ax.axvline(x=date_num, color='red', linestyle='--', linewidth=0.5)
            # if event_labels:
            #     # Add event labels as annotations
            #     ax.annotate(event_labels[i], xy=(date_num, ax.get_ylim()[1]), xytext=(date_num, ax.get_ylim()[1] + 20),
            #                 textcoords='offset points', ha='center', color='red', fontsize=9, rotation=45)


        # Plot lines and scatter points for each segment using indexing
        for i in range(len(dmr)):
            segment = dmr[i]
            start = segment['start']
            end = segment['end']
            models = segment['model']
            
            # Convert start and end dates to matplotlib date numbers
            time_range = mdates.date2num([start, end])
            
            # Access the corresponding time_series DataFrame from the list
            time_series = self.domain_segments[i]
            print(time_series.columns)
            for key, value in models.items():
                coeffs = value.params
                
                # Using point-slope form to calculate corresponding y values
                start_val = self.domain_mapping[start]
                end_val = self.domain_mapping[end]
                
                # Calculate y-values using the linear model: y = b0 + b1 * time
                start_y = coeffs[0] + coeffs[1] * start_val
                end_y = coeffs[0] + coeffs[1] * end_val
                
                # Plot the line segment
                ax.plot([start, end], [start_y, end_y], color=color_map[key], label=key)
                
                # Extract the last three digits of the key, convert to float
                column_name = float(key[-3:])
                if column_name in time_series.columns:
                    # Randomly thin the points by selecting a random subset
                    sample_size = int(len(time_series) * 0.1)  # Adjust the percentage as needed
                    thinned_time_series = time_series.sample(n=sample_size, random_state=42)
                    
                    # Plot scatter points with smaller size and transparency
                    ax.scatter(thinned_time_series.index, thinned_time_series[column_name], 
                            color=color_map[key], marker=marker_map[key], s=10, alpha=0.6, label=f"{key} data")

                if report:
                    print(f"Segment Start: ({start}, {start_val}), Segment End: ({end}, {end_val}), Model: {key}")
                    print(f"Beta Coefficients: {coeffs}")
                    print("\n")

        # Set axis labels
        ax.set_xlabel('Time (Hours)')
        ax.set_ylabel('X-Pixel Location')


        # Create a secondary x-axis for top ticks aligned with the unique dates
        ax2 = ax.twiny()
        
        # Ensure the top axis has unique and properly formatted date labels
        unique_date_nums = list(dict.fromkeys(date_nums))  # Remove duplicates while preserving order
        ax2.set_xticks(unique_date_nums)
        
        # Convert date numbers back to readable date labels
        formatted_labels = [mdates.num2date(num).strftime('%Y-%m-%d') for num in unique_date_nums]

        # Apply formatted date labels to the top axis with red color and bold font
        ax2.set_xticklabels(formatted_labels, ha='center', 
                            fontdict={'color': 'red', 'weight': 'bold'})

        # Synchronize the bounds of the upper x-axis with the lower axis
        ax2.set_xlim(ax.get_xlim())

        # Position the ticks of the top x-axis at the top
        ax2.xaxis.set_ticks_position('top')
            
        # Handle legend to avoid duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        print(f"Handles: {handles}")
        print(f"Labels: {labels}")
        unique_labels = dict(zip(labels, handles))
        
        if len(dmr[0]['model']) > 1:
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(1.2, 1))
        
        
        if ax is None:
            ax.set_title('Segmented Domain Shoreline Trend Models')
            plt.show()

    # create a model report for the subplots
    def make_model_report(self, ax):
        report_lines = []

        for i, segment in enumerate(self.domain_model_results):
            
            start = segment['start']
            start_date = start.date()            
            end_date = segment['end'].date()
            report_lines.append(f"Date Range: {start_date} to {end_date}")
            
            # Access the corresponding time_series DataFrame from the list
            time_series = self.domain_segments[i]
            print(f"Columns in Time Series: {time_series.columns}")
            
            # Compute the variance for the specific column (199.0)
            if 199.0 in time_series.columns:
                specific_variance = time_series[199.0].var()
                specific_std_dev = np.sqrt(specific_variance)  # or directly use .std()
                print(f"Variance for 199.0: {specific_variance}")
                print(f"Standard Deviation for 199.0: {specific_std_dev}")
            else:
                print("Column 199.0 not found in the time series.")
            # # segment variance
            # segment_variance = np.sqrt(time_series.var())
            # print(f"Var Type{type(segment_variance)}")
            
            for key, model in segment['model'].items():
                coeffs = model.params
                beta0 = coeffs[0]
                beta1 = coeffs[1]
                start_val = self.domain_mapping[start]
                beta0_start = beta0 + beta1 * start_val
                
                # report_lines.append(f"  Model: {key}")
                # report_lines.append(r"Variance ($\sigma^2$): {:.4f}".format(segment_variance))
                report_lines.append(f"Standard Deviation: {specific_std_dev:.2f}")
                report_lines.append(r"Intercept ($\beta_0$): {:.2f}".format(beta0_start))
                report_lines.append(r"Trend ($\beta_1$)(pixels/hour): {:.2f}".format(coeffs[1]))
                report_lines.append(r"Trend ($\beta_1$)(pixels/month) {:.2f}".format(coeffs[1] * (24*30)))

                report_lines.append("")

        report_text = "\n".join(report_lines)
        
        # Display the report text in the specified subplot (ax)
        ax.text(0.1, 0.9, report_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left')
        # ax.axis('off')  # Hide axes for the text report
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.title.set_text('Model Report')


    def plot_all_models(self):
        # Create a figure with a custom grid layout using gridspec
        fig = plt.figure(figsize=(20, 6))
        
        # Define the grid layout with 1 row and 2 columns
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio (75% and 25%)
        
        # Create the subplots within the grid layout
        ax1 = fig.add_subplot(gs[0])  # This will be 75% of the width
        ax2 = fig.add_subplot(gs[1])  # This will be 25% of the width
        
        # Plot in the first subplot
        self.plot_segmented_domain_model(ax=ax1)
        
        # Plot the report in the second subplot
        self.make_model_report(ax=ax2)
        
        # Adjust layout to prevent overlap
        plt.suptitle('Segmented Domain Shoreline Trend Models', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
        

    def segment_boxplots(self, ax=None):
        n_segments = len(self.domain_segments)
        
        if ax is None:
            # Create a grid of subplots with n rows and 3 columns
            n_rows = (n_segments + 2) // 3  # Ensure enough rows for all segments
            fig, axs = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
            axs = axs.flatten()  # Flatten the grid for easy indexing
        else:
            # Use the provided ax in an n x 3 grid
            axs = [ax] * n_segments
        
        # Load the image
        avg_image_path = './images/oakisland_west/timex/oakisland_west-2023-12-01-160842Z-timex.jpeg'
        image = Image.open(avg_image_path)

        # Resize the image to 30% of its original dimensions
        new_size = (int(image.width * 0.3), int(image.height * 0.3))
        resized_image = image.resize(new_size, Image.ANTIALIAS)

        # Convert the resized image to a NumPy array
        resized_image_array = np.array(resized_image)

        # Subset the new image pixels to be that of the shoreline transects domain
        subset_image_array = resized_image_array[160:240, 0:495, :]

        # Loop over each segment and create boxplots
        for i, (segment, ax_i) in enumerate(zip(self.domain_segments, axs)):
            # Overlay the image within the boxplot grid
            dmr = self.domain_model_results
            start = dmr[i]['start'].date()
            end = dmr[i]['end'].date()
            ax_i.imshow(subset_image_array, aspect='auto', extent=[0, 1, 0, 1], alpha=0.5, zorder=-1, transform=ax_i.transAxes)

            # Reverse the columns and create boxplots on the same axis
            segment.iloc[:, ::-1].boxplot(ax=ax_i, vert=False, patch_artist=True)

            # Customize the subplot
            ax_i.set_title(f'{start} to {end}')
            ax_i.set_xlabel('Detected Shoreline Pixel X-coordinates')
            ax_i.set_ylabel('Transect Y-coordinates')

        # Final plot adjustments
        plt.tight_layout()
        plt.suptitle('Detected Shoreline Transect Boxplots', fontsize=16, y=1.02)
        
        if ax is None:
            plt.show()

###########################################################################################################################


class ShorelineStats:
    def __init__(self, df):
        self.df = df.copy()
        self.results = None
    
    def perform_regression(self):
        """
        Perform row-wise linear regression where each row represents the x coordinates
        and the column names represent the y coordinates, with the y-axis reversed.
        """
        results = []
        metrics = {
            'intercept': [],
            'slope': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'conf_int_low': [],
            'conf_int_high': [],
            'p_values': [],
            't_values': [],
            'std_errors': []
        }
        
        for index, row in self.df.iterrows():
            x = row.values
            y = self.df.columns.astype(float).values[::-1]  # Reverse the y-axis

            # Add a constant (intercept) to the model
            X = sm.add_constant(y)
            
            # Perform regression
            model = sm.OLS(x, X).fit()
            results.append(model)
            
            # Collect metrics
            predictions = model.predict(X)
            metrics['intercept'].append(model.params[0])
            metrics['slope'].append(model.params[1])
            metrics['rmse'].append(np.sqrt(mean_squared_error(x, predictions)))
            metrics['mae'].append(mean_absolute_error(x, predictions))
            metrics['r2'].append(model.rsquared)
            conf_int = model.conf_int()
            metrics['conf_int_low'].append(conf_int[1, 0])
            metrics['conf_int_high'].append(conf_int[1, 1])
            metrics['p_values'].append(model.pvalues[1])
            metrics['t_values'].append(model.tvalues[1])
            metrics['std_errors'].append(model.bse[1])
        
        self.results = results
        return pd.DataFrame(metrics, index=self.df.index)
    
    def merge_metrics_with_df(self, metrics_df):
        merged_df = self.df.copy()
        for column in metrics_df.columns:
            merged_df[column] = metrics_df[column].values
        return merged_df
    
    def plot_shoreline_regressions(self, row_index):
        """
        Perform regression for a specific row using the perform_regression function,
        and plot the results, including all metrics and the date from the datetime index.
        """
        # Perform regression on the entire dataset
        metrics_df = self.perform_regression()
        
        # Get the model and metrics for the specified row
        model = self.results[row_index]
        predictions = model.predict(sm.add_constant(self.df.columns.astype(float).values[::-1]))
        date = self.df.index[row_index]

        # Metrics to include in the plot
        metrics = {
            'Intercept': 'intercept',
            'Slope': 'slope',
            'RMSE': 'rmse',
            'MAE': 'mae',
            'R^2': 'r2',
            'Conf. Int. Low': 'conf_int_low',
            'Conf. Int. High': 'conf_int_high',
            'P-value': 'p_values',
            'T-value': 't_values',
            'Std. Error': 'std_errors'
        }

        # Extract the relevant data for the specified row
        x = self.df.iloc[row_index].values
        y = self.df.columns.astype(float).values[::-1]

        # Plot the data and the regression line
        plt.figure(figsize=(10, 6))
        plt.scatter(y, x, label='Observed Data', color='blue')
        plt.plot(y, predictions, label='Fitted Line', color='red')
        plt.title(f'Regression for Date: {date}')
        plt.xlabel('Y-axis (Reversed)')
        plt.ylabel('X values')
        plt.legend()

        # Annotate the plot with metrics
        metrics_text = "\n".join([f'{name}: {metrics_df.loc[date, key]:.4f}' for name, key in metrics.items()])
        
        plt.gca().text(0.05, 0.5, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5),
                    transform=plt.gca().transAxes)
        
        plt.show()


