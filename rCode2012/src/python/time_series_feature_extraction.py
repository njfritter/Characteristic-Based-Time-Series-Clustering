# Python script containing core time series feature extraction functions (translated from R code)

### GENERAL IMPORTS ###
import numpy as np
import pandas as pd
import math

### SIGNAL PROCESSING SPECIFIC IMPORTS ###
# For estimating the frequency of a given time series
from scipy.signal import periodogram
# For decomposing the time series
from scipy.stats import boxcox
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.gam.smooth_basis import BSplines,CyclicCubicSplines
import statsmodels.api as sm
# For measure calculation
from scipy.special import inv_boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox
from pypr.stattest.ljungbox import boxpierce

### RPY2 SPECIFIC IMPORTS (SO WE CAN IMPLEMENT R CODE) ###
# Requires some manual work before these code blocks will work
# 1. Set R_HOME variable (can be found by typing ".Library" into R Studio)
# 2. The output from the above is the "library" subdirectory of the R_HOME directory established by R Studio
# 3. Take the output minus "/library" at the end, and save this to R_HOME in a terminal
# 4. Run pip3 install rpy2 (connects to R_HOME at install)
# Will make an init.sh file to run all the necessary steps for this + other manual setup steps

# Sources: https://stackoverflow.com/questions/17573988/r-home-error-with-rpy2
# https://stackoverflow.com/questions/47585718/rpy2-installed-but-wont-run-packages
# https://stackoverflow.com/questions/24880493/how-to-find-out-r-library-location-in-mac-osx/24880594
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri, numpy2ri, Formula
from rpy2.robjects.vectors import IntVector,FloatVector
# Load necessary R packages
rtseries = importr('tseries')
rbase = importr('base')
rstats = importr('stats')
rfracdiff=importr('fracdiff')
rutils=importr('utils')
rmgcv=importr('mgcv')

# "Activate" pandas2ri and numpy2ri
#pandas2ri.activate()
#numpy2ri.activate()

# For some reason, running the above commands blocks me from being able to create R time series objects 
# Trying to create time series objects just leads to numpy arrays being created
# https://www.r-bloggers.com/using-r-in-python-for-statistical-learning-data-science-2/
# Example output when the below commands are NOT used:
'''
rbase.set_seed(123) # reproducibility seed
x = r.ts(r.rnorm(n=10)) # simulate the time series
print(x)

Time Series:
Start = 1 
End = 10 
Frequency = 1 
 [1] -0.56047565 -0.23017749  1.55870831  0.07050839  0.12928774  1.71506499
 [7]  0.46091621 -1.26506123 -0.68685285 -0.44566197
'''

# Example output when the below commands are used:
'''
rbase.set_seed(123) # reproducibility seed
x = r.ts(r.rnorm(n=10)) # simulate the time series
print(x)

[-0.56047565 -0.23017749  1.55870831  0.07050839  0.12928774  1.71506499
  0.46091621 -1.26506123 -0.68685285 -0.44566197]
'''


def find_freq(x):
    # Use an iterative function to automagically determine the frequency of the time series data
    # Takes in a single column of a pandas DataFrame as a univariate time series
    
    n = len(x)
    # Now estimate the spectral density of the time series via AR fit
    # Two ways: numpy fft method or scipy signal method
    # Method #1: numpy fft method (https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python)
    '''
    pow_np = np.abs(np.fft.fft(x))**2
    time_step = 1 / n
    freqs = np.fft.fftfreq(n, time_step)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps[idx])
    '''
    
    # Method #2: scipy signal method via density or spectrum (takes an optional second frequency parameter)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
    #f, pow_scipy = signal.periodogram(x,scaling='density')
    f, pow_scipy = periodogram(x,scaling='spectrum')
    
    # Iterate through frequencies 
    freq = f
    power = pow_scipy
    if max(power) > 10: # Arbitrary threshold chosen by trial and error.
        # The power might be a huge number way of out the index bounds, so pick max index if so
        power_idx = min(int(round(max(power))),(len(power)-1))
        freq_idx = min(int(round(power[power_idx])),len(freq)-1)
        period = 1/freq[freq_idx]
        # If period is infinity, find next local maximum
        if period == np.Inf:
            j = pd.Series(power).diff() > 0
            if len(j) > 0:
                nextmax = j[1] + power[int(round(max(power[1:])))]
                if (nextmax <= len(freq)):
                    period = int(round(1/freq[int(round(nextmax))]))
                else:
                    period = 1
            else:
                period = 1
        else:
            period = int(round(period))
    else:
        period = 1
    
    return period

def find_freq_r(x):
    # Same as above function, but using the R code directly via rpy2
    n = len(x)
    spec = rstats.spec_ar(na_contiguous(x),plot=False)
    spec_vals = spec[np.where(spec.names == 'spec')[0].item()]
    spec_freq = spec[np.where(spec.names == 'freq')[0].item()]

    if max(spec_vals) > 10:
        #period <- round(1/spec$freq[which.max(spec$spec)])
        period = round(1/spec_freq[rbase.which_max(spec_vals)[0] - 1])
        if period == np.Inf: # Find next local maximum
            #j <- which(diff(spec$spec)>0)
            j = rbase.which(rbase.diff(spec_vals) > 0)
            if len(j) > 0:
                #nextmax <- j[1] + which.max(spec$spec[j[1]:500])
                nextmax = j[0] + rbase.which_max(spec_vals[int(j[0]):500])
                #print(nextmax.item())
                if nextmax.item() - 1 <= len(spec_freq):
                    period = round(1/spec_freq[nextmax.item() - 1])
                else:
                    period = 1
            else:
                period = 1
    else:
        period = 1
    
    return int(period)

def na_contiguous(x):
    # Recreate na.contiguous function in R since this is used frequently
    # This takes a series object with a time index and finds the longest consecutive stretch of non-missing values
    # https://stackoverflow.com/questions/41494444/pandas-find-longest-stretch-without-nan-values
    # And then return the shortened dataframe with all non-null values
    values = x.values 
    mask = np.concatenate(( [True], np.isnan(values), [True] ))  # Mask
    start_stop = np.flatnonzero(mask[1:] != mask[:-1]).reshape(-1,2)   # Start-stop limits
    start,stop = start_stop[(start_stop[:,1] - start_stop[:,0]).argmax()]  # Get max interval, interval limits
    contiguous = x.iloc[start:stop]
    return contiguous

def decompose(x, transform = True):
    # Decompose data into trend, seasonality and randomness
    # Accepts a pandas series object with a datetime index
    if (transform and min(x.dropna()) >= 0):
        # Transforms data and finds the lambda that maximizes the log likelihood 
        # R version has above method and method that minimizes the coefficient of variation ("guerrero")
        x_transformed, var_lambda = boxcox(na_contiguous(x),lmbda = None)
        x_transformed = pd.Series(x_transformed,index=na_contiguous(x).index)
    
    else:
        x_transformed = x
        var_lambda = np.nan
        transform = False
        
    # Seasonal data 
    # In R code, we find the number of samples per unit time below (should be 1 every time)
    # Here I take the datetime index differences, take their inverses, and store in a list to be evaluated
    # https://stackoverflow.com/questions/36583859/compute-time-difference-of-datetimeindex
    idx = x_transformed.index
    #samples = np.unique([int(1/(idx[n]-idx[n - 1]).days) for n in range(1,len(idx))])
    # Filter out Nulls for this exercise
    #samples = samples[~np.isnan(samples)]
    #if len(samples) == 1 and samples.item() > 1:

    # Just use the R code instead
    # This is supposed to be "> 1" but all data results in a frequency of 1
    # All frequency results in R equal 4, meaning this code block gets evaluated every time in R
    # So this code block should always be evaluated as well
    if int(rstats.frequency(x_transformed).item()) == 1:
        # Decompose
        stl = sm.tsa.seasonal_decompose(na_contiguous(x_transformed))
        #stl = rstats.stl(na_contiguous(x_transformed),s_window='periodic')
        # When I try to use above function, I get this:
        '''
        R[write to console]: Error in (function (x, s.window, s.degree = 0, t.window = NULL, t.degree = 1,  : 
  series is not periodic or has less than two periods
        '''
        trend = stl.trend
        seasonality = stl.seasonal
        remainder = x_transformed - trend - seasonality

    else:
        # Nonseasonal data
        trend = pd.Series(np.nan, index=x_transformed.index)
        time_index = pd.Index([i for i in range(1,len(x_transformed)+1)])
        # Python specific
        bs = BSplines(time_index, df=[12, 10], degree=[3, 3])
        cs = CyclicCubicSplines(time_index,df=[3,3])
        alpha = np.array([218.338888])
        gam = GLMGam(x_transformed, smoother=cs, alpha=alpha).fit()
        #trend.loc[~x_transformed.isnull()] = gam.fittedvalues
        
        # R Code
        fmla = Formula('x ~ s(tt)')
        env = fmla.environment
        env['tt'] = time_index
        env['x'] = x_transformed
        trend.loc[~x_transformed.isnull()] = rstats.fitted(rmgcv.gam(fmla))
        seasonality = pd.Series(np.nan, index=x_transformed.index)
        remainder = x_transformed - trend
    
    return_dct = {
        'x': x_transformed,
        'trend': trend,
        'seasonality': seasonality,
        'remainder': remainder,
        'transform': transform,
        'lambda': var_lambda,
    }
    
    return return_dct

def f1_transformation(x, a, b):
    eax = math.exp(a * x)
    if eax == np.Inf:
        f1_eax = 1
    else:
        f1_eax = (eax-1)/(eax+b)
    return f1_eax

def f2_transformation(x, a, b):
    eax = math.exp(a*x)
    ea = math.exp(a)
    return((eax-1)/(eax+b)*(ea+b)/(ea-1))

def calculate_measures(x):
    # Save ts version of our data for some of the below functions
    #rbase.set_seed(123) # reproducibility seed
    #x_ts_contiguous = r.ts(FloatVector(na_contiguous(x)))
    #print(x_ts_contiguous)
    
    # Now "activate" pandas2ri and numpy2ri
    pandas2ri.activate()
    numpy2ri.activate()
    
    N = len(x)
    freq = find_freq_r(x)
    fx = (math.exp((freq-1)/50)-1)/(1+math.exp((freq-1)/50))
    
    # Decomposition
    decomp_x = decompose(x)
    
    # Adjust data
    # Unfortunately it looks like frequency is calculated a different way in the decompose function
    # Thus there may be data for which this function is evaulated when 'seasonality' is null
    # Going to add an extra check to make sure to not evaluate this if all the values are null
    #print(decomp_x['seasonality'])
    if freq > 1 and (not decomp_x['seasonality'].isnull().all()):
        fit = decomp_x['trend'] + decomp_x['seasonality']
    else:
        # Nonseasonal data
        fit = decomp_x['trend']
    adj_x = decomp_x['x'] - fit + np.mean(decomp_x['trend'].dropna())
    
    # Backtransformation of adjusted data
    if decomp_x['transform']:
        # The below line of code doesn't work for some reason
        #t_adj_x = inv_boxcox(adj_x.values, decomp_x['lambda'])
        # Use actual formula instead (but do inverse because we're solving for x)
        '''
        The Box-Cox transform is given by:

            y = (x**lmbda - 1) / lmbda,  for lmbda > 0
                log(x),                  for lmbda = 0
        '''
        if decomp_x['lambda'] == 0:
            # Assuming base of 10 (x = 10^y)
            t_adj_x = 10 ** adj_x
        else:
            # x = ((y * lambda) + 1) ^ (1/lambda)
            t_adj_x = ((adj_x * decomp_x['lambda']) + 1) ** (1/decomp_x['lambda'])
    else:
        t_adj_x = adj_x
    
    # Trend and seasonal measures
    v_adj = np.var(adj_x.dropna())
    threshold = 0.00000000001
    if(freq > 1):
        detrend = decomp_x['x'] - decomp_x['trend']
        deseason = decomp_x['x'] - decomp_x['seasonality']
        
        if np.var(deseason.dropna()) < threshold:
            trend = 0
        else:
            trend = max(0,min(1,1-(v_adj/np.var(deseason.dropna()))))
        if np.var(detrend.dropna()) < threshold:
            seasonality = 0
        else:
            seasonality = max(0,min(1,1-(v_adj/np.var(detrend.dropna()))))
    else:
        # Nonseasonal data
        if np.var(decomp_x['x'].dropna()) < threshold:
            trend = 0
        else:
            trend = max(0,min(1,1-(v_adj/np.var(decomp_x['x'].dropna()))))
        seasonality = 0
    
    measures = [fx,trend,seasonality]
    
    # Measures on original data
    xbar = np.mean(x.dropna())
    std = np.std(x.dropna())
    
    # Serial correlation (make sure box pierce statistic is returned as well)
    #bp = boxpierce(x, lags=max_lag)
    #Had to fix stattest module in pypr package via: https://gist.github.com/betterxys/1def38e1fcbb7f3b2dab2393bcea52f0
    max_lag = 10
    lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(x, lags=max_lag, boxpierce=True)
    # The above returns values for each lag, so just grab the final value
    Q = bpvalue[-1] / (N*max_lag)
    fQ = f2_transformation(Q,7.53,0.103)
    
    # Nonlinearity (THIS REQUIRES THE TIMESERIES OBJECT VERSION OF OUR DATA)
    '''
    non_linear_test = rtseries.terasvirta_test_ts(x_ts_contiguous,type = "Chisq")
    #non_linear_test = rtseries.terasvirta_test_default(y=x_contiguous,x=x_contiguous.index.dayofyear,type = "Chisq")
    p = non_linear_test[np.where(non_linear_test.names == 'statistic')[0].item()][0]
    fp = f1_transformation(p,0.069,2.304)
    '''
    fp = None
    
    # Skewness
    skew = abs(np.mean((x.dropna()-xbar) ** 3)/std ** 3)
    fs = f1_transformation(skew,1.510,5.993)
    
    # Kurtosis
    kurtosis = np.mean((x.dropna()-xbar) ** 4)/std ** 4
    fk = f1_transformation(kurtosis,2.273,11567)
    
    # Hurst=d+0.5 where d is fractional difference
    hurst = rfracdiff.fracdiff(na_contiguous(x),0,0)
    H = hurst[np.where(hurst.names == 'd')[0].item()].item() + 0.5 
    
    # Lyapunov Exponent
    if freq > (N-10):
        # There is insufficient data, declare this variable as none
        fLyap = None
    else:
        Ly = np.zeros(N-freq)
        for i in range(0,(N-freq)):
            diffs = abs(x.iloc[i] - x)
            date_idx = diffs.sort_values().index
            int_idx = pd.Index([diffs.index.get_loc(date) for date in date_idx])
            idx = int_idx[int_idx < (N-freq)]
            j = idx[1]
            try:
                Ly[i] = math.log(abs((x.iloc[i+freq] - x.iloc[j+freq])/(x.iloc[i]-x.iloc[j]))) / freq
            except ValueError: # domain error, means log(0) was taken
                Ly[i] = 0
            if(np.isnan(Ly[i]) or (Ly[i] == np.Inf) or (Ly[i] == -np.Inf)):
                Ly[i] = np.nan
        Lyap = np.mean(Ly[~np.isnan(Ly)])
        fLyap = math.exp(Lyap) / (1+math.exp(Lyap))
    
    measures = measures + [fQ,fp,fs,fk,H,fLyap]
    
    # Measures on adjusted data
    xbar = np.mean(t_adj_x.dropna())
    std = np.std(t_adj_x.dropna())

    # Serial correlation (make sure box pierce statistic is returned as well)
    #bp = boxpierce(adj_x, lags=max_lag)
    max_lag = 10
    lbvalue, pvalue, bpvalue, bppvalue = acorr_ljungbox(na_contiguous(adj_x), lags=max_lag, boxpierce=True)
    # The above returns values for each lag, so just grab the final value
    Q = bpvalue[-1] / (N*max_lag)
    fQ = f2_transformation(Q,7.53,0.103)

    # Nonlinearity (add try/except block to capture data where this doesn't work)
    # (THIS REQUIRES THE TIMESERIES OBJECT VERSION OF OUR DATA)
    try:
        adj_x_contiguous = na_contiguous(adj_x)
        non_linear_test = rtseries.terasvirta_test_ts(adj_x_contiguous,type = "Chisq")
        #non_linear_test = rtseries.terasvirta_test_default(y=adj_x_contiguous,x=adj_x_contiguous.index.dayofyear,type = "Chisq")
        p = non_linear_test[np.where(non_linear_test.names == 'statistic')[0].item()][0]
        fp = f1_transformation(p,0.069,2.304)
    except ValueError:
        print('This block did not work for the following data:\n',adj_x)
    
    # Skewness
    skew = abs(np.mean((t_adj_x.dropna() - xbar) ** 3)/(std ** 3))
    fs = f1_transformation(skew,1.510,5.993)

    # Kurtosis
    kurtosis = np.mean((t_adj_x.dropna() - xbar) ** 4)/(std ** 4)
    fk = f1_transformation(kurtosis,2.273,11567)
    
    measures_list = measures + [fQ,fp,fs,fk]

    measures_df = pd.DataFrame.from_dict(measures_dct,orient='index',columns=["frequency", "trend","seasonal", "autocorrelation","non-linear","skewness","kurtosis","Hurst","Lyapunov","dc autocorrelation","dc non-linear","dc skewness","dc kurtosis"])

    return measures_df