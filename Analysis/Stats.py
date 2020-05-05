from __future__ import division

import sys 
import warnings
sys.path.append(r'M:\Python\Personal\DataAnalysis')
sys.path.append(r'C:\Development\Git\DataAnalysis')

import numpy as np
import pandas as pd

from Analysis import Plotting
from ParallelComputing import Compute, Functions

import statsmodels as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scipy.stats as st
from scipy.stats import kurtosis
from scipy.stats import skew

def anova(data, group):
    """group should be   """
    print(pairwise_tukeyhsd(data, group))


def BoxPlotStats(data):
    """
    Calculate statistics for use in violin plot.
    """
    x = np.asarray(data, np.float)
    vals_min = np.min(x)
    vals_max = np.max(x)
    q2 = np.percentile(x, 50, interpolation="linear")
    q1 = np.percentile(x, 25, interpolation="lower")
    q3 = np.percentile(x, 75, interpolation="higher")
    iqr = q3 - q1
    whisker_dist = 1.5 * iqr

    # in order to prevent drawing whiskers outside the interval
    # of data one defines the whisker positions as:
    d1 = np.min(x[x >= (q1 - whisker_dist)])
    d2 = np.max(x[x <= (q3 + whisker_dist)])
    extra = len(x[x > d2])/len(x)

    std = np.std(x)
    var = np.var(x)
    kurtosis_ = kurtosis(x)
    skew_ = skew(x)
    
    return {
        "std": std,
        "var": var,
        "kurtosis": kurtosis_,
        "skew" : skew_,
        "iqr": iqr,
        "min": vals_min,
        "max": vals_max,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "d1": d1,
        "d2": d2,
        "extra": extra
    }

def Correlations(df, mic = False):
    #note that Mic and Tick are in ParallelComputing.Functions

    df = df.select_dtypes(include=['int64','float64'])

    pearson  = df.corr(method = "pearson")
    spearman = df.corr(method = "spearman")
    kendall  = df.corr(method = "kendall")

    if mic:
        output = Compute.asyncDFListProcessing(Functions.Maximal_Compute, df, df.columns)
        mic = pd.DataFrame(output).set_index('index')
    else:
        mic = None

    return {"Pearson": pearson, "Spearman": spearman, "Kendall": kendall, "MICe": mic}

def EDA(df):
    
    Plotting.ColumnDetails(df)
    Plotting.DistributionSeries([df[column].values for column in df.columns], y_names = list(df.columns))
    Plotting.PairwiseDensityPlot(df)
    Plotting.CorrelationHeatmap(df)
    Plotting.AndrewsCurve(df)
    Clustering.HDBScan(df)
    
    return


#Basic Baysian Inference:
#https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = scipy.stats._continuous_distns._distn_names
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000):
    """
       Generate distributions's Propbability Distribution Function '
	   Check available distributions via scipy.stats._continuous_distns._distn_names
    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def make_cdf(dist, params, size=10000):
    """
    Generate distributions's Cumulative Propbability Distribution Function 
	Check available distributions via scipy.stats._continuous_distns._distn_names
    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.cdf(x, loc=loc, scale=scale, *arg)
    cdf = pd.Series(y, x)

    return cdf