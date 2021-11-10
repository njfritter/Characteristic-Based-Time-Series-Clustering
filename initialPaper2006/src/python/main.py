# Example script that shows a run through of the code I implemented 
# From the original 2006 paper
# Import all needed functions from timeSeriesFeatureExtraction.py script

# General imports
import pandas as pd
import numpy as np

# Code specific imports
from timeSeriesFeatureExtraction import time_series_raw_feature_extraction,transform_raw_features_into_final_features,breakout_chaos_values_and_combine,linear_transformation,softmax_scaling

# Import data for Treasury yield rates
treasury_yield_rates_df = pd.read_csv('../../data/treasuryYieldRates.csv',sep=",",header=0,parse_dates=['Date'])
treasury_yield_rates_df['Weekday'] = treasury_yield_rates_df['Date'].dt.day_name()
print(treasury_yield_rates_df.head(5))
print(treasury_yield_rates_df.columns)
print(treasury_yield_rates_df.dtypes)
# Add indicator column that we will use as our index
treasury_yield_rates_df['Indicator'] = 'Tresuary'

# Import data for S&P 500 Open/Close/High/Low/Volume Numbers
sp_500_df = pd.read_csv('../../data/SP500FinancialData.csv',sep=",",header=0,parse_dates=['Date'])
#sp_500_df['Date'] = pd.to_datetime(sp_500_df['Date'],format='%m/%d/%Y')
print(sp_500_df.head(5))
print(sp_500_df.columns)
print(sp_500_df.dtypes)

# Declare variables that we want to analyze
date_col = 'Date' # DO NOT CHANGE
index_col = '6 mo' # Column or other feature we want to analyze
analysis_df = treasury_yield_rates_df

# Declare some numbers for input to the feature extraction function
# FEEL FREE TO CHANGE; the below values are also the default values for each input

# Variables to help filter out data + fill in missing data
fill_function = np.mean # Function that we will use to fill in missing data (ONLY NUMPY FUNCTIONS)
fill_percentile = None # USE ONLY IF ABOVE IS "np.percentile" AND WISH TO FILL IN DATA VIA PERCENTILES
days_of_data = 92 # Need to think of better way of filling this in
days_of_missing_data = 7 # Max days of missing data we are willing to accept

# Feature extraction parameters (below values are defaults to the below function)
le_time_periods = 10 # Number of time periods we look ahead for Lyapunov Exponent calculation
trans_parameter = 0.500 # Parameter with which we transform the raw data
max_time_lag = 20 # Maximum time lag considered for autocorrelation caculations
smoothing_factor = 1222 # Positive smoothing factor used to choose number of knots for regression spline (used to transform raw into detrended/de-seasonalzed data)
min_adjustment_val = 0.001 # For transformed data if minimum series value is zero, add this number multiplied by the maximum series value to each value

#### BEGIN ANALYSIS ####

## Get raw time series features ##
# There is an OPTIONAL visualization analysis that can to be done as an intermediate step
# Before the final features can be obtained
time_series_raw_feature_dct = time_series_raw_feature_extraction(analysis_df,index_col,date_col,days_of_data,days_of_missing_data,fill_function,fill_percentile,le_time_periods,trans_parameter,max_time_lag,smoothing_factor,min_adjustment_val)
'''
## Get final non-standardized time series features ##
# FEATURES STILL TO BE EXTRACTED: Self-similiarity (raw), Non-linearity (raw/TSA)
time_series_processed_feature_dct = transform_raw_features_into_final_features(time_series_raw_feature_dct)

## Break out chaos values into their own columns ##
# This will also return a df for later use
non_standardized_processed_feature_df = breakout_chaos_values_and_combine(time_series_processed_feature_dct)

## Standardize data ##
# The paper has a special standardization method, but they rely on specific feature ranges (all nonnegative)
# Decided on other recommended methods stated in paper (either linear transform or softmax scaling)
linearly_transformed_df = linear_transformation(non_standardized_processed_feature_df)
softmax_scaled_df = softmax_scaling(non_standardized_processed_feature_df)
'''