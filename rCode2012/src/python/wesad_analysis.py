'''
Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven. 2018. 
Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. 
In 2018 International Conference on Multimodal Interaction (ICMI '18), October 16-20, 2018, Boulder, CO, USA. ACM, New York, NY, USA, 9 pages.
'''
# Example script analyzing Wearable Stress and Affect Detection (WESAD) Dataset from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29#

# Import needed packages
import matplotlib # Do this first
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np

# Import function to get WESAD data from our central data_etl.py script
from data_etl import get_processed_wesad_data

# Import time series feature extraction method
from time_series_feature_extraction import calculate_measures
    
if __name__ == '__main__':
    # Perform analysis
    subject_dct = get_processed_wesad_data()


    measures_df = calculate_measures(df)