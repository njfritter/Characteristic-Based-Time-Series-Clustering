# This code is my attempt at turning the original 2006 paper on time series clustering into Python code
# Even though some of my attempted calculations ended up quite off from the results of the R code posted in 2012, they generated decent clusters ()

### VISUALIZATION SPECIFIC IMPORTS ###
# Doing this first to avoid errors
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

### GENERAL IMPORTS ###
import pandas as pd
import numpy as np
import sys

### SIGNAL PROCESSING SPECIFIC IMPORTS ###
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
from scipy.signal import detrend
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from math import log,floor,ceil

### CLUSTERING SPECIFIC IMPORTS ###
# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram

# Self Organizing Map
# Going to use the approach from https://github.com/hhl60492/SOMPY_robust_clustering/blob/master/sompy/examples/main.py
# As well as apply k means clustering to the clusters generated by the above
# Also had to do some workarounds to get sompy package to work: https://github.com/sevamoo/SOMPY/issues/36
# Needed to do the following to get this to work:
# pip3 install git+https://github.com/compmonks/SOMPY.git
# pip3 install ipdb==0.8.1
from sompy.sompy import SOMFactory
from sompy.visualization.mapview import View2D
from sompy.visualization.umatrix import UMatrixView
from sompy.visualization.hitmap import HitMapView
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Feature extraction parameters (below values are defaults to the below functions)
# Feel free to play around with these and see how the clustering results are affected
le_time_periods = 10 # Number of time periods we look ahead for Lyapunov Exponent calculation
trans_parameter = 0.500 # Parameter with which we transform the raw data
max_time_lag = 20 # Maximum time lag considered for autocorrelation caculations
smoothing_factor = 1222 # Positive smoothing factor used to choose number of knots for regression spline (used to transform raw into detrended/de-seasonalzed data)
min_adjustment_val = 0.001 # For transformed data if minimum series value is zero, add this number multiplied by the maximum series value to each value

## Clustering parameters
# Hierarchical Clustering
n_clusters = 3 # Number of clusters for clustering
linkage = 'complete' # Type of hierarchical clustering method
affinity = 'euclidean' # "Proximity" Measure for hierarchical clustering

# Self Organizing Map parameters
normalization = 'var'
initialization = 'pca'
n_job = 2
verbose = 'info'
train_rough_len = 1
train_finetune_len = 5

### DATA PROCESSING FUNCTIONS
# NOTE: For these functions, they will assume that the passed data contains a 
# date column (this code hasn't been tested with time columns) 
# and an index column for different categories of time series that we are comparing
# I.e. company name to compare stock prices, baseball player name to compare batting average, etc.
def get_unique_indices(data,index_col):
    
    return data[index_col].unique()

def compute_mean_median_and_plot(data,date_col):
    # First grab all columns 
    data_cols = data.columns
    data_mean_median = pd.DataFrame(data.groupby([date_col]).agg(
        {
         col: {col + '_MEAN': 'mean', col + '_MEDIAN': 'median'},
         } for col in data_cols
    ).reset_index().to_records()).drop(columns='index')
    
    # Set column names before plotting
    
    data_mean_median.set_index(date_col,inplace=True)
    data_mean_median.plot(subplots=True,legend=True,figsize=(15,15))

def filter_out_indices(data,index_col,date_col,days_of_data,days_of_missing_data,col):
    # This function filters out indices that have more than "days_of_missing_data" missing in the "col" column
    num_unique_indices_before = len(get_unique_indices(data,index_col))
    data_ind = data.groupby([index_col]).agg({col: 'count'}) > (days_of_data - days_of_missing_data)
    data_ind_filtered = data_ind.loc[data_ind[col] == True,:].reset_index()
    data_filtered = data.loc[data[index_col].isin(data_ind_filtered[index_col])].sort_values(by=[index_col,date_col])
    num_unique_indices_after = len(get_unique_indices(data_filtered,index_col))
    print('%s of %s indices filtered out' % (num_unique_indices_before - num_unique_indices_after, num_unique_indices_before))
    
    return data_filtered

def fill_missing_values(data,index_col,function,percentile=None):
    # This function will take data with missing values and fill them according to the function passed (mean, median, etc.)
    # This must be done before the raw feature extraction, as we need full data points for this to work
    # The function passed MUST be an actual Python function (i.e. mean(), median(), np.percentile, etc)
    filled_data = pd.DataFrame(columns=data.columns)
    for index in get_unique_indices(data,index_col):
        subset_df = data.loc[data[index_col] == index,:]
        if percentile:
            subset_df = subset_df.fillna(function(subset_df,percentile))
        else:
            subset_df = subset_df.fillna(function(subset_df))
        filled_data = pd.concat([filled_data,subset_df])
    return filled_data

def create_df_subset_dict(data,index_col):
    # Turns dataframe into dict of Index column-Data combinations
    df_subset_dict = {}
    data_unique_ids = get_unique_ids(data,index_col)
    for i in range(0,len(data_unique_ids)):
        df_subset = data.loc[data[index_col] == data_unique_ids[i],:]
        df_subset_dict[i] = df_subset
    
    return df_subset_dict

# This function was to be used to estimate the frequency of the time series data
# Haven't gotten it working yet
'''
def analysis_peaks(sample_data):
    peaks,_ = find_peaks(sample_data,distance=20)
    peaks2,_ = find_peaks(sample_data,prominence=0.3)
    peaks3,_ = find_peaks(sample_data,width=3)
    peaks4,_ = find_peaks(sample_data,threshold=0.2)

    plt.subplot(2, 2, 1)
    plt.plot(np.array(peaks), sample_data[peaks], "xr"); plt.plot(sample_data); plt.legend(['distance'])

    plt.subplot(2, 2, 2)
    plt.plot(peaks2, sample_data[peaks2], "ob"); plt.plot(sample_data); plt.legend(['prominence'])
    plt.subplot(2, 2, 3)
    plt.plot(peaks3, sample_data[peaks3], "vg"); plt.plot(sample_data); plt.legend(['width'])
    plt.subplot(2, 2, 4)
    plt.plot(peaks4, sample_data[peaks4], "xk"); plt.plot(sample_data); plt.legend(['threshold'])

    plt.show()

'''

# This function was to be used to estimate the Hurst parameter using FARIMA
# Don't think there is a Python package for FARIMA
'''
def estimate_hurst_parameter():
    import itertools
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 365) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                results = model.fit()
                print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
'''

### FEATURE EXTRACTION FUNCTIONS ###
def extract_raw_features(df_subset_dict,index_col,analysis_column,date_col,le_time_periods=10,trans_parameter=0.5,max_time_lag=20,smoothing_factor=1222,min_adjustment_val=0.001):
    # Takes in a df_subset_dict created by the "create_df_subset_dict" method above and extracts raw features
    # Assumes exactly ONE feature is being inputted for analysis
    # Could probably refactor to do as multiple, but for now it will be one variable at a time
    # This for loop already takes some time to complete, so multiple variables may be unwieldy
        
    # Begin analysis
    analysis_time_series_features_raw_dct = {}
    for index,df_subset in df_subset_dict.items():
        # Extract unique category name for each set of data
        index = df_subset[index_col].unique().item()

        # Set the activity date as the index and convert the activity date into a datetime index
        # So that the seasonal decomposition function to transform the data later works
        df_subset[date_col] = pd.to_datetime(df_subset[date_col])
        analysis_df = df_subset[[date_col,analysis_column]].set_index(date_col)
        # Convert analysis data to numeric data in case data is "object" datatype
        analysis_df[analysis_column] = pd.to_numeric(analysis_df[analysis_column],errors='coerce')

        ##### RAW FEATURE PRE-PROCESSING ######
        # Process raw data for feature calculation later (we will perform the transformation needed for the TSA features below)
        # Skew & Kurtosis Calculations (These are direct calculations, so these are simpler)
        analysis_skew_raw = skew(analysis_df[analysis_column])
        analysis_kurtosis_raw = kurtosis(analysis_df[analysis_column])
        
        ### Autocorrelation Pre-processing ###
        # Because this involves comparing the original data with various time-lagged versions of the data
        # We need to iteratively calculate many different autocorrelation series (Qh = n∑hk=1 rk^2)
        # Where n is the length of the time series, k is the current lag considered and h is the maximum time lag being considered (~20)
        # We will store the results of this in dictionaries to be saved into the final dictionary
        h = max_time_lag
        analysis_autocorrelation_raw_dct = {'LENGTH': len(analysis_df), 'VALUES': {}}
        for k in range(1, h+1):
            analysis_autocorrelation_raw = pd.concat([analysis_df,analysis_df.shift(k)],axis=1).corr().iloc[0,1] # Could be [1,0] too
            analysis_autocorrelation_raw_dct['VALUES'].update({k : analysis_autocorrelation_raw})
        ### The "chaos" metric (Lyapunov Exponent) Pre-processing ###
        # The paper says that for this metric, for all i = 1,2,...,n Xj should be the "nearest" point to Xi
        # I am taking that definition to mean that Xj is the next consecutive point to Xi (i.e. j = i + 1)
        # It also says that |Xj - Xi| should be small; for the most part this appears to be true with how I've defined it
        # So we will take the difference of each consecutive point (|X2 - X1|, |X3 - X2|, etc.) as a new dataframe
        # And then take the quotient of each point at varying points in the future to the current point
        # I.e. |Xj+n - Xi+n| / |Xj - Xi| with n the number of time points looking ahead (the le_time_periods variable above)
        # The final formula for each individual point is (1/n) * log(|Yj+n − Yi+n|/|Yj − Yi|)
        analysis_diff = analysis_df.diff(periods=1)
        analysis_lyapunov_exponents_dct = {}
        for n in range(1, le_time_periods + 1):
            analysis_lyapunov_exponents = abs(analysis_diff) / abs(analysis_diff.shift(n))
            analysis_lyapunov_exponent = analysis_lyapunov_exponents[analysis_column].apply(lambda x: (1 / n) * log(x) if x > 0 and x < float('Inf') else np.nan)
            analysis_lyapunov_exponents_dct[n] = analysis_lyapunov_exponent

        ### Periodicity (frequency) Pre-processing ###
        # First need detrended data (original data minus a fitted regression spline with 3 knots)
        # Then need all autocorrelation values on detrended data for all lags up to 1/3 the series length
        # Then from the autocorrelation function above, we examine it for peaks and troughs (in the next code block)
        # Frequency is the first peak provided with following conditions:
        # (a) there is also a trough before it
        # (b) the difference between peak and trough is at least 0.1
        # (c) the peak corresponds to positive correlation
        # If no such peak is found, frequency is set to 1 (equivalent to non-seasonal)
        analysis_x,analysis_y = analysis_df.reset_index().index.values,analysis_df.values
        analysis_reg_spline = UnivariateSpline(x=analysis_x,y=analysis_y,s=smoothing_factor)
        analysis_reg_spline_values = analysis_reg_spline.__call__(analysis_x)
        analysis_detrended = analysis_y.transpose() - analysis_reg_spline_values

        # Autocorrelation values on detrended data
        max_lag = int(ceil((1/3) * analysis_detrended.shape[1]))
        analysis_detrended_autocorrelation_values = []
        analysis_detrended = pd.DataFrame(analysis_detrended.transpose())
        for k in range(0,max_lag):
            analysis_detrended_autocorrelation = pd.concat([analysis_detrended,analysis_detrended.shift(k)],axis=1).corr().iloc[0,1] # Could be [1,0] too
            analysis_detrended_autocorrelation_values.append(analysis_detrended_autocorrelation)

        ##### TRANSFORMED FEATURE PRE-PROCESSING ######
        # In order to get the TSA (trend & seasonality adjusted) features, we need to assess whether the minimum values are non-negative
        # We only consider a transformation if theminimum of{Yt}is non-negative. If the minimum ofYtis zero, we add a small posi-tive constant (equal to 0.001 of the maximum ofYt) to all values to avoid undefinedresults 
        min_analysis = min(analysis_df[analysis_column])
        if min_analysis == 0:
            max_analysis = max(analysis_df[analysis_column])
            analysis_df[analysis_column] = analysis_df[analysis_column] + (min_adjustment_val * max_analysis)

        # Perform transformation Yt* = (Yt^λ − 1)/λ where λ != 0 and λ ∈ (−1,1)
        analysis_transformed = (pow(analysis_df[analysis_column],trans_parameter) - 1)/trans_parameter

        # Decompose the transformed data to get TSA data for processing
        analysis_decomposition = sm.tsa.seasonal_decompose(analysis_transformed,model='additive',extrapolate_trend='freq')
        # Calculate features based on tsa data
        # Trend & Seasonality Calculations

        ### Skew & Kurtosis Pre-processing ###
        analysis_skew_tsa = skew(analysis_decomposition.resid)
        analysis_kurtosis_tsa = kurtosis(analysis_decomposition.resid)
        ### Autocorrelation Pre-processing ###
        # Because this involves comparing the original data with various time-lagged versions of the data
        # We need to iteratively calculate many different autocorrelation series (rk=Corr(Yt,Yt−k) with many k values)
        # Later for actual feature calcuation, we will be using the Box-Pierce statistic (Qh = n∑hk=1 rk^2)
        # Where n is the time series length, k is the current lag considered and h is the maximum time lag being considered (~20)
        # We will store the results of this in dictionaries to be processed later and stored in the final dictionary
        h = max_time_lag
        analysis_autocorrelation_tsa_dct = {'LENGTH': len(analysis_df), 'VALUES': {}}
        for k in range(1,h+1):
            analysis_autocorrelation_tsa = pd.concat([analysis_decomposition.resid,analysis_decomposition.resid.shift(k)],axis=1).corr().iloc[0,1]
            analysis_autocorrelation_tsa_dct['VALUES'].update({k : analysis_autocorrelation_tsa})

        analysis_time_series_features_raw_dct[index] = {
            'ANALYSIS_DECOMPOSITION': analysis_decomposition,
            'ANALYSIS_SKEW_RAW': analysis_skew_raw,
            'ANALYSIS_KURTOSIS_RAW': analysis_kurtosis_raw,
            'ANALYSIS_SKEW_TSA': analysis_skew_tsa,
            'ANALYSIS_KURTOSIS_TSA': analysis_kurtosis_tsa,
            'ANALYSIS_AUTOCORRELATION_TSA': analysis_autocorrelation_tsa_dct,
            'ANALYSIS_AUTOCORRELATION_RAW': analysis_autocorrelation_raw_dct,
            'ANALYSIS_DIFF': analysis_diff,
            'ANALYSIS_LE_VALUES_DCT': analysis_lyapunov_exponents_dct,
            'ANALYSIS_DETRENDED_AUTOCORRELATION': analysis_detrended_autocorrelation_values,
        }
        
    return analysis_time_series_features_raw_dct

def transform_raw_features_into_final_features(raw_features_dct):
    # This function takes the analysis_time_series_features_raw_dct output from the extract_raw_features method above
    # And applies transformations to give us the non-standardized features to be used in our clustering algorithm
    analysis_time_series_features_processed_dct = {}
    for index,raw_feature_dct in raw_features_dct.items():

        ###### RAW DATA FEATURE CALCULATIONS #######
        ### Autocorrelation feature will be Box-Pierce statistic (Qh = n∑hk=1 rk^2) ###

        # Box-Pierce statistic (Autocorrelation)
        analysis_df_length_raw = raw_feature_dct['ANALYSIS_AUTOCORRELATION_RAW']['LENGTH']
        analysis_inner_sum_raw = sum([pow(value,2) for key,value in raw_feature_dct['ANALYSIS_AUTOCORRELATION_RAW']['VALUES'].items()])
        analysis_box_pierce_stat_raw = analysis_df_length_raw * analysis_inner_sum_raw

        ### Non-linearity (Teraesvirta Neural Network Test for Nonlinearity) ###
        # This can be done according to the R package here: http://math.furman.edu/~dcs/courses/math47/R/library/tseries/html/terasvirta.test.html
        # Could not find anything in Python, but luckily the rpy2 package allows for use of R packages within Python
        # Guide to proper installation below
        # https://bitbucket.org/rpy2/rpy2/issues/403/cannot-pip-install-rpy2-with-latest-r-340
        pass

        ### Skewness & Kurtosis (already calculated above) ###
        analysis_skew_raw = raw_feature_dct['ANALYSIS_SKEW_RAW']
        analysis_kurtosis_raw = raw_feature_dct['ANALYSIS_KURTOSIS_RAW']

        ### Self-similiarity (using of fractional ARIMA to measure long range dependence) ###
        pass

        ### Lyapunov Exponent Calculation (Measure of "chaos") ###
        analysis_lyapunov_exponents_n = {n: np.mean(lyapunov_exponents) for n,lyapunov_exponents in raw_feature_dct['ANALYSIS_LE_VALUES_DCT'].items()}

        ### Periodicity (measure of frequency) ###
        # Need to use detrended data and serial correlation values for all lags up to 1/3 of series length
        # Did this and the raw_feature_dct being used does have these values
        # However pretty much all of them don't have peaks that meet the requirements listed above
        # So will set the periodicity for all data to 1 for now (equivalent to non-seasonal)
        analysis_periodicity_raw = 1

        ###### TSA DATA FEATURE CALCULATIONS #######

        ### TREND & SEASONALITY ###
        analysis_trend_tsa = 1 - (np.var(raw_feature_dct['ANALYSIS_DECOMPOSITION'].resid) / np.var(raw_feature_dct['ANALYSIS_DECOMPOSITION'].seasonal))
        analysis_seasonality_tsa = 1 - (np.var(raw_feature_dct['ANALYSIS_DECOMPOSITION'].resid) / np.var(raw_feature_dct['ANALYSIS_DECOMPOSITION'].trend))

        ### Box-Pierce statistic (Autocorrelation) ###
        analysis_df_length_tsa = raw_feature_dct['ANALYSIS_AUTOCORRELATION_TSA']['LENGTH']
        analysis_inner_sum_tsa = sum([pow(value,2) for key,value in raw_feature_dct['ANALYSIS_AUTOCORRELATION_TSA']['VALUES'].items()])
        analysis_box_pierce_stat_tsa = analysis_df_length_tsa * analysis_inner_sum_tsa

        ### Non-linearity (Teraesvirta Neural Network Test for Nonlinearity) ###
        pass

        ### Skewness & Kurtosis (already calculated above) ###
        analysis_skew_tsa = raw_feature_dct['ANALYSIS_SKEW_TSA']
        analysis_kurtosis_tsa = raw_feature_dct['ANALYSIS_KURTOSIS_TSA']

        ##### ADD PROCESSED FEATURES TO FINAL DICTIONARY #####
        analysis_time_series_features_processed_dct[index] = {
            'ANALYSIS_BOX_PIERCE_STAT_RAW': analysis_box_pierce_stat_raw,
            #'ANALYSIS_NON_LINEARITY_RAW': ,
            'ANALYSIS_SKEW_RAW': analysis_skew_raw,
            'ANALYSIS_KURTOSIS_RAW': analysis_kurtosis_raw,
            #'ANALYSIS_SELF_SIM_RAW': ,
            'ANALYSIS_CHAOS_MEASURE_RAW': analysis_lyapunov_exponents_n,
            'ANALYSIS_PERIODICTY_RAW': analysis_periodicity_raw,
            'ANALYSIS_TREND_TSA': analysis_trend_tsa,
            'ANALYSIS_SEASONALITY_TSA': analysis_seasonality_tsa,
            'ANALYSIS_BOX_PIERCE_STAT_TSA': analysis_box_pierce_stat_tsa,
            #'ANALYSIS_NON_LINEARITY_TSA': ,
            'ANALYSIS_SKEW_TSA': analysis_skew_tsa,
            'ANALYSIS_KURTOSIS_TSA': analysis_kurtosis_tsa,
        }
    return analysis_time_series_features_processed_dct

def breakout_chaos_values_and_combine(processed_features_dct):
    # Takes in a dictionary object outputted by the "transform_raw_features_into_final_features" method above
    # Turns the dict into a df and breaks out data from the chaos measure field, which is a nested dictionary
    # Adds the new fields to the df and removes the old nested dict column
    
    # Turn original dictionary into a dataframe first
    processed_features_df = pd.DataFrame.from_dict(processed_features_dct,orient='index',dtype=float)
    # Luckily, we can use a neat one liner to break out the nested dictionary; keys -> column names, values -> column values
    # https://stackoverflow.com/questions/38231591/splitting-dictionary-list-inside-a-pandas-column-into-separate-columns
    analysis_chaos_values = processed_features_df['ANALYSIS_CHAOS_MEASURE_RAW'].apply(pd.Series)
    new_columns = [('ANALYSIS_CHAOS_MEASURE_RAW_' + col) for col in analysis_chaos_values.columns.astype(str)]
    analysis_chaos_values.columns = new_columns
    
    # Combine the broken out data with the original data, drop the original nested column
    processed_features_combined_df = pd.concat([processed_features_df,analysis_chaos_values],axis=1).drop(columns='ANALYSIS_CHAOS_MEASURE_RAW')
    return processed_features_combined_df

### SCALING FUNCTIONS
def linear_transformation(df):
    # Take in an input dataframe, take each value in every column and apply the linear transformation column wise
    # Because pandas operates column wise, we can use the following one liner
    linear_transformed_data = ((df - df.min())/(df.max() - df.min()))
    if linear_transformed_data['ANALYSIS_PERIODICTY_RAW'].isnull().sum() == len(linear_transformed_data):
        # This means all values got coerced to nulls; will set back to 1 here
        print('Converting null values back to 1')
        linear_transformed_data['ANALYSIS_PERIODICTY_RAW'] = 1

    return linear_transformed_data

def softmax_scaling(df):
    # Take in an input dataframe, take each value in every column and apply the softmax transformation column wise
    # Because pandas operates column wise, we can use the following one liner
    #softmax_scaled_data = df.apply(lambda x: 1 / (1 + math.exp(-pd.to_numeric(x))))
    softmax_scaled_data = pd.DataFrame()
    for col in df.columns:
        softmax_scaled_data_col = df[col].apply(lambda x: (1 / (1 + math.exp(-x))))
        softmax_scaled_data = pd.concat([softmax_scaled_data,softmax_scaled_data_col],axis=1)
    return softmax_scaled_data

### CLUSTERING FUNCTIONS ###
# Hierarchical Clustering
def hierarchical_clustering_sklearn(normalized_df,n_clusters=2,affinity='euclidean',linkage='complete'):
    clustering = AgglomerativeClustering(n_clusters=n_clusters,affinity=affinity,linkage=linkage).fit(normalized_df)
    print('Number of clusters: ',clustering.n_clusters_)
    print('Number of leaves: ',clustering.n_leaves_)
    print('Number of connected components: ',clustering.n_connected_components_)
    return clustering

def hierarchical_clustering_scipy(normalized_df,method='complete',metric='euclidean'):
    clustering = sch.linkage(normalized_df,method=method,metric=metric)
    return clustering

def plot_dendrogram_scipy(clustering_model):
    dendrogram = sch.dendrogram(clustering_model)
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

def plot_dendrogram_sklearn(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

# Self Organizing Map
def self_organizing_map(normalized_df,normalization='var',initialization='pca',n_job=1,train_rough_len=2,train_finetune_len=5,verbose=None):
    # create the SOM network and train it. You can experiment with different normalizations and initializations
    som = SOMFactory().build(normalized_df.values,normalization=normalization,initialization=initialization,component_names=normalized_df.columns)
    som.train(n_job=n_job,train_rough_len=train_rough_len,train_finetune_len=train_finetune_len,verbose=verbose)
    
    # The quantization error: average distance between each data vector and its BMU.
    # The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
    topographic_error = som.calculate_topographic_error()
    quantization_error = np.mean(som._bmu[1])
    print("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))
    return som

def som_component_planes(som):
    # component planes view
    view2D = View2D(15,15,"rand data",text_size=12)
    view2D.show(som, col_sz=4, which_dim="all", desnormalize=True)
    plt.show()

def som_kmeans_clustering_predict(som,k):
    # This performed K-means clustering with k clusters on the SOM grid to PREDICT clusters
    #[labels, km, norm_data] = som.cluster(K,K_opt)
    map_labels = som.cluster(n_clusters=k)
    data_labels = np.array([map_labels[int(k)] for k in som._bmu[0]])
    hits = HitMapView(20,20,"Clustering",text_size=12)
    a=hits.show(som)
    return som,map_labels

def som_kmeans_clustering(som,k):
    # Perform K Means clustering on the SOM grid just FITTING data so we can get more data returned
    kMeansCluster = KMeans(n_clusters=k).fit(
        som._normalizer.denormalize_by(som.data_raw,
                                        som.codebook.matrix))
    return kMeansCluster

def som_u_matrix(som):
    # U-matrix plot
    umat = UMatrixView(width=10,height=10,title='U-matrix')
    umat.show(som)

def plot_elbow_curve(X,som,method,K_values=range(1,10)):
    # Method via https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    # Takes a SOM grid, trains a K Means clustering algorithm using the different K_values
    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    for k in K_values: 
        # Training a KMeans clustering model using the transformed SOM grid
        kMeansClustering = som_kmeans_clustering(som,k=k)
        # Save distortion and inertia values
        distortions.append(sum(np.min(cdist(X, kMeansClustering.cluster_centers_, 
                          'euclidean'),axis=1)) / X.shape[0]) 
        inertias.append(kMeansClustering.inertia_) 

        mapping1[k] = sum(np.min(cdist(X, kMeansClustering.cluster_centers_, 
                     'euclidean'),axis=1)) / X.shape[0] 
        mapping2[k] = kMeansClustering.inertia_
        
        # Save SSE values
        # Compute the L2-norm of the vector difference between each element in cluster n and cluster n’s centroid, and add this to the total SSE
        # The L2 norm that is calculated as the square root of the sum of the squared vector values
        
        ### ADD CODE FOR SSE CALCULATION HERE ###
    
    # Plot elbow curve according to supplied method (either 'distortion' or 'inertia')
    if method == 'inertia':
        plt.plot(K_values, inertias, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Inertia') 
        plt.title('The Elbow Method using Inertia') 
    if method == 'distortion':
        plt.plot(K_values, distortions, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Distortion') 
        plt.title('The Elbow Method using Distortion') 
    if method == 'sse':
        pass
        
    plt.show()

### CLUSTER VISUALIZATION FUNCTIONS ###
# https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

def concat_data_and_clusters(X,index_col,features,labels):
    indices_and_cluster = pd.concat([pd.Series(features.index),pd.Series(labels)],axis=1)
    indices_and_cluster.columns = [index_col,'CLUSTER']
    x_with_clusters = pd.merge(X,indices_and_cluster,how='inner',on=index_col)
    return x_with_clusters

def visualize_cluster(original_time_series_data,time_series_features_processed_df,clustering):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    x_with_clusters = concat_data_and_clusters(original_time_series_data,time_series_features_processed_df,clustering.labels_)
    
    # Create dictionaries for storing results/
    clusters = {}
    summaries = {}
    histograms = {}
    for cluster in x_with_clusters['CLUSTER'].unique():
        current_cluster = x_with_clusters.loc[x_with_clusters['CLUSTER'] == cluster]
        current_cluster_summary = current_cluster.describe()
        current_cluster_histogram = current_cluster.hist(bins=50,figsize=(8,8))
        print(current_cluster_summary)
        clusters[cluster] = current_cluster
        summaries[cluster] = current_cluster_summary
        histograms[cluster] = current_cluster_histogram

def plot_data_with_clusters(original_time_series_data,time_series_features_processed_df,clustering,analysis_column,date_col):
    x_with_clusters = concat_data_and_clusters(original_time_series_data,time_series_features_processed_df,clustering.labels_)
    
    # According to docs, can plot labeled data this way if the labels have the column name 'y'
    '''
    x_with_clusters['y'] = x_with_clusters['CLUSTER']
    x_col = 'ACTIVITY_DT'
    y_col = analysis_column[0]
    plt.plot(x_col,y_col,data=x_with_clusters)
    plt.show()
    '''
    x = x_with_clusters[analysis_column]
    dates = x_with_clusters[[date_col]]
    labels = x_with_clusters['CLUSTER']
    all_colors = ['red','green','blue','orange','purple']
    colors = all_colors[:len(set(labels))]
    #plt.scatter(x,y,c=labels)
    plt.scatter(dates.values.ravel().tolist(),x.values.ravel().tolist(),c=labels.values.ravel(),cmap=plt.cm.Spectral)
    plt.show()
    '''
    xi = x_with_clusters[['ACTIVITY_DT']]
    yi = x_with_clusters[analysis_col]
    labels = x_with_clusters[['CLUSTER']]
    plt.plot(xi, yi, labels=labels)
    #plt.legend()
    plt.show()
    
    by_label = tmp_df.groupby('CLUSTER')
    for name, group in by_label:
        plt.plot(group['ACTIVITY_DT'], group[analysis_col[0]], label=name)

    plt.legend()
    plt.show()
    '''

def plot_2d_clusters(original_time_series_data,time_series_features_processed_df,clustering,columns=[]):
    x_with_clusters = concat_data_and_clusters(original_time_series_data,time_series_features_processed_df,clustering.labels_)
    
    if columns:
        # https://stackoverflow.com/questions/42056713/matplotlib-scatterplot-with-legend
        x = x_with_clusters[columns[0]]
        y = x_with_clusters[columns[1]]
        unique = list(set(labels))
        colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
        for i, u in enumerate(unique):
            xi = [x[j] for j  in range(len(x)-1) if labels[j] == u]
            yi = [y[j] for j  in range(len(x)-1) if labels[j] == u]
            plt.scatter(xi, yi, c=colors[i], label=str(u))
        plt.legend()
        plt.show()
    '''
    
    x=[1,2,3,4]
    y=[5,6,7,8]
    classes = [2,4,4,2]
    unique = list(set(classes))
    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
    for i, u in enumerate(unique):
        xi = [x[j] for j  in range(len(x)) if classes[j] == u]
        yi = [y[j] for j  in range(len(x)) if classes[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u))
    plt.legend()

    plt.show()
    '''
def plot_cluster_aggregate_values(original_df,transformed_df,cluster_labels,date_col):
    x_with_clusters = concat_data_and_clusters(original_df,transformed_df,cluster_labels)
    by_label_model = x_with_clusters.groupby('CLUSTER')
    for name, group in by_label_model:
        by_activity_date = group.groupby(date_col)
        cluster_aggs = {}
        for date, inner_group in by_activity_date:
            cluster_mean = inner_group[analysis_col].median()
            cluster_aggs[date] = cluster_mean
        plt.plot(list(cluster_aggs.keys()),list(cluster_aggs.values()),label='Softmax cluster %s' % str(name))

    plt.legend()
    plt.show()

### DO IT ALL FUNCTIONS ###
def time_series_raw_feature_extraction(univariate_ts_data,index_col,date_col,days_of_data,days_of_missing_data=30,fill_function=np.mean,fill_percentile=None,le_time_periods=10,trans_parameter=0.5,max_time_lag=20,smoothing_factor=1222,min_adjustment_val=0.001):
    # This function executes the subset of the above functions needed to do the time series feature extraction + clustering
    # So it make take a while for it all to complete, so will add print statements to show completion of steps
    # Takes in specifically a univariate time series (as designated above) as input
    # and has an index and date column
    
    # Get unique index values
    data = univariate_ts_data
    unique_indices = get_unique_indices(data,index_col)
    
    # Get analysis column
    index_and_date_cols = [index_col,date_col]
    analysis_col = data.columns[~data.columns.isin(index_and_date_cols)].values[0]
    # Filter out time series with less than inputted # of days of missing data
    filtered_data = filter_out_indices(data,date_col,days_of_data,days_of_missing_data,analysis_col)
    # And then fill in missing data via a passed function
    filtered_filled_data = fill_missing_values(filtered_data,analysis_col,fill_function,percentile=None)
    # Create subset dictionary where each time series is a value
    filtered_filled_subset_dict = create_df_subset_dict(filtered_filled_data)

    # Run the first feature extraction function (takes the longest time)
    initial_raw_feature_dct = extract_raw_features(filtered_filled_subset_dict,analysis_col,date_col,le_time_periods,trans_parameter,max_time_lag,smoothing_factor,min_adjustment_val)
    return initial_raw_feature_dct

# Main function for use to input univariate time series data and return time series features
def main(analysis_data,index_col,date_col,days_of_data,days_of_missing_data,fill_function,fill_percentile,le_time_periods,trans_parameter,max_time_lag,smoothing_factor,min_adjustment_val):
    
    ## Get raw time series features ##
    # There is an OPTIONAL visualization analysis that can to be done as an intermediate step
    # Before the final features can be obtained
    time_series_raw_feature_dct = time_series_raw_feature_extraction(analysis_data,index_col,date_col,days_of_data,days_of_missing_data,fill_function,fill_percentile,le_time_periods,trans_parameter,max_time_lag,smoothing_factor,min_adjustment_val)

    ## Get final non-standardized time series features ##
    # FEATURES STILL TO BE EXTRACTED: Self-similiarity (raw), Non-linearity (raw/TSA)
    time_series_processed_feature_dct = transform_raw_features_into_final_features(time_series_raw_feature_dct)

    ## Break out chaos values into their own columns ##
    # This will also return a df for later use
    # non_standardized_processed_feature_df = pd.DataFrame.from_dict(time_series_processed_feature_dct,orient='index',dtype=float)
    non_standardized_processed_feature_df = breakout_chaos_values_and_combine(time_series_processed_feature_dct)

    ## Standardize data ##
    # The paper has a special standardization method, but they rely on specific feature ranges (all nonnegative)
    # Decided on other recommended methods stated in paper (linear transformation and softmax scaling)
    linearly_transformed_df = linear_transformation(non_standardized_processed_feature_df)
    softmax_scaled_df = softmax_scaling(non_standardized_processed_feature_df)
    
    return time_series_raw_feature_dct,time_series_processed_feature_dct,non_standardized_processed_feature_df,linearly_transformed_df,softmax_scaled_df

