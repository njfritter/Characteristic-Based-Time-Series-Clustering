# Characteristic Based Time Series Clustering Analysis

This work is inspired by the following paper ([link to paper on Rob's website](https://robjhyndman.com/publications/ts-clustering/) and [link to Researchgate article](https://www.researchgate.net/publication/220451959_Characteristic-Based_Clustering_for_Time_Series_Data):

```
"Characteristic-based clustering for time series data"
Xiaozhe Wang, Kate A Smith, Rob J Hyndman
(2006) Data Mining and Knowledge Discovery 13(3), 335-364
```

## My Work

In this repo I will be showcasing my work in attempting to turn the above paper into Python code for general use in extracting features from time series data and using them as inputs to various time series clustering methods.

I realized after beginning that the R code had been linked in the following blog [post](https://www.r-bloggers.com/measuring-time-series-characteristics/). I have also turned this code into Python for comparison to my original code.

## Automatic Time Series Feature Extraction Packages

More recently, it has come to my attention that there are various R packages that do automatic feature extraction from time series data: the [tsfeatures package](https://pkg.robjhyndman.com/tsfeatures/articles/tsfeatures.html) and the [feasts package](https://github.com/tidyverts/feasts) (intending to replace the tsfeatures package). There have been similar efforts in Python, with the [tsfresh Python package](https://github.com/blue-yonder/tsfresh) currently being developed in parallel to the R package work. 

These automatic feature extraction packages will be used in conjunction with my custom feature extraction functions and their cluster effectiveness will be compared.

## Future Work

I hope to leverage these automatic feature extraction packages (and perhaps my custom scripts) to try and cluster together some time series data that is interesting to me:

  - Sports data, such as 3-point field goal % and points per game (PPG) for basketball, batting average and runs per game for baseball, TDs and turnovers for football, etc.
  - Financial data, such as popular blue chip stocks and index funds
  - Health and clinical data, especially with the COVID-19 situation worldwide

Will update this repo as needed.