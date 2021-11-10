# Example script that shows a run through of the new tsfresh package
# Source code here: https://github.com/blue-yonder/tsfresh
# Read the docs: https://tsfresh.readthedocs.io/en/latest/

# General imports
import pandas as pd
import numpy as np

# Code specific imports
from tsfresh import extract_features,select_features
from tsfresh.feature_extraction import feature_calculators

# Import data for Treasury yield rates
treasury_yield_rates_df = pd.read_csv('../../data/treasuryYieldRates.csv',sep=",",header=0,parse_dates=['Date'])
treasury_yield_rates_df['Weekday'] = treasury_yield_rates_df['Date'].dt.day_name()
print(treasury_yield_rates_df.head(5))
print(treasury_yield_rates_df.columns)
print(treasury_yield_rates_df.dtypes)
# Add indicator column that we will use as our index
treasury_yield_rates_df['Indicator'] = 'Treasury'

# Import data for S&P 500 Open/Close/High/Low/Volume Numbers
sp_500_df = pd.read_csv('../../data/SP500FinancialData.csv',sep=",",header=0,parse_dates=['Date'])
sp_500_df['Date'] = pd.to_datetime(sp_500_df['Date'],format='%m/%d/%Y',errors='coerce')
# This creates nulls since there are rows with "#########" as the date value
# Let's just filter these out for now
sp_500_df = sp_500_df.loc[~sp_500_df['Date'].isnull()]
print(sp_500_df.head(5))
print(sp_500_df.columns)
print(sp_500_df.dtypes)

# Declare variables that we want to analyze
date_col = 'Date' # DO NOT CHANGE
index_col = '6 mo' # Column or other feature we want to analyze
analysis_df = treasury_yield_rates_df