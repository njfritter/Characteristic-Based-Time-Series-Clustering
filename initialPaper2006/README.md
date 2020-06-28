# Initial Paper on Characteristic Based Time Series Clustering (2006)

Clustering has always been an intriguing part of data science; watching a visualization of a simple k-Nearest Neighbor (kNN) algorithm start with randomized cluster center locations and iteratively update and end up with intuitive cluster locations and similar data points grouped together is a mezmorizing feat.

With more and more time series data being generated and collected by companies and institutions there are now more opportunities than ever to analyze this data and generate new insights that previously weren't possible.

This data is by nature unlabeled, which means that cluster labels would need be generated in some way. Manual labeling of data is fine but also time and cost intensive. Using unsupervised machine learning methods to generate groupings can be a huge time saver but could also lead to strange results that are not human interpretable. Regardless, this can allow for all sorts of intriguing insights into how clustering algorithms might "decide" on cluster groupings.

At first look, trying to cluster the actual time series data itself seems non-trivial:
  - How would one calculate Euclidean distance (a common measure calculated in clustering algorithms) on one time series versus another?
  - How would one handle missing data?
  - How would one deal with time series of different frequencies (i.e. daily vs hourly) and length (1 year vs 3 years)?
  - Might time series with vastly different absolute values or units still be similar in other ways?

Clearly clustering the raw time series data presents its own set of problems that are not straightforward to solve. So how might we go about clustering this data together in order to extract something meaningful?

## Time Series Feature Extraction

Typically part of time series analysis involves decomposing the original time series into its seasonal, trend and remainder (or random) components. This is a key process in trying to make conclusions about what the data is doing over the period of time of the time series:
  - Is the data increasing or decreasing over time? (Trend)
  - Does the data repeat itself every x number of time periods? (Seasonality)

These can be thought of as *characteristics* of the time series data that summarize different aspects of the behavior of the time series. The great part about them is that they are more robust for missing values and can be extracted for time series of any length (although the longer the time series, the better the signal extracted).

The makers of the paper realized this and decided to try to cluster together *extracted features* from time series data. This would allow for comparison of time series of different shapes and sizes while using features that summarized time series data quite effectively.

In order to try and capture as much behavior from the data as possible, 13 measures are extracted from the time series. Some measures are calculated on the raw data, and some on what is called "TSA" (trend & seasonality adjusted) data. Details on all the features are given below.

### Breaking Down the Features

The features extracted from the time series data is are as follows:
 




## Scaling (Standardizing) the Raw Features 

After extracting the final features but before inputting them into a clustering algorithm, we must now scale them into the [0,1] range so that features with large absolute values do not dominate the clustering.

A statistical transformation will be used to map each raw feature Q to rescaled q with a range of [0,1].

Depending on the range of Q, we will apply one of the following transformations:
1. If raw feature Q has a range of [0,∞], we rescale using 
\begin{equation*} q = \frac{(e^{aQ}−1)}{(b+e^{aQ})} \end{equation*} 
where a and b are constants to be chosen
    - This measure is referred to as f1
2. If raw feature Q has a range of [0,1], we rescale using
\begin{equation*} q = \frac{(e^{aQ}−1)(b+e^{a})}{(b+e^{aQ})(e^{a}−1)} \end{equation*}
    - We choose a and b such that q satisfies the conditions:
        - q has 90th percentile = 0.10 when Yt is standard normal white noise, and q has value of 0.9 for a well-known benchmark dataset with the required feature. 
    - This is referred to as f2
3. Lastly, if raw feature Q has a range of [1,∞], we rescale using 
\begin{equation*} q = \frac{(e^{\frac{Q−a}{b}}−1)}{(1+e^{\frac{Q−a}{b}})} \end{equation*}
where a and b are constants to be chosen, with q satisfying the conditions: 
    - q=0.1 for Q=12 and q=0.9 for Q=150.
    - This last transformation is referred to as f3

Since the scaling methods mentioned above required the unscaled data to be nonnegative (and some of the raw features I ended up with were negative) I decided to go with alternative methods.

## Scaling (Standardizing) the Raw Features via Alternative Methods

Alternative effective scaling methods were also mentioned in the paper (albeit whose purpose was to be be used for comparison in clustering experiments later in the paper):
  - A linear transform method, mapping (–∞,∞) to [0,1] range:
\begin{equation*} vnorm = \frac{v_i − min(v_1...v_n)}{max(v_1...v_n)−min(v_1...v_n)} \end{equation*}
    - Where vnorm is the normalized value and v_i is a instance value of actual values
  - The Softmax scaling (the logistic function) to map (–∞,∞) to (0, 1):
\begin{equation*} v_n = \frac{1}{1+e^{−v_i}} \end{equation*}
where \begin{equation*} e^{−v_i} = 1/e^{v_i} \end{equation*} v_n denotes the normalized value and v_i denotes a instance value of actual values 