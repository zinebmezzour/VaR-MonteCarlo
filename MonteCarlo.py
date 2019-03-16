
from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from pyspark import SparkContext, SparkConf
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from datetime import timedelta
from itertools import islice
import statsmodels.api as sm
from os import listdir
from os.path import isfile, join
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import math
import statistics
import itertools

import sys
from operator import add
from pyspark.sql import SparkSession


portfolio = pd.read_csv('spark/portfolio.csv', sep=';')
factors = pd.read_csv('spark/all_factors.csv.2', sep=';')

portfolio.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
factors.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

#fill factors Nan Values 
factors.fillna(method='bfill', inplace = True)
factors.fillna(method='ffill', inplace = True)

factors_2=factors[['4. close', 'squared', 'root', 'nasdaq_close', 'squared_nq' , 'root_nq', 'Value', 'squared_oil', 'root_oil', '1 YR', 'squared_tr', 'root_tr']].pct_change(4)
pf = portfolio.loc[:, portfolio.columns != 'Date']
pf_2 = pf.pct_change(4)
pf_2['Date'] = portfolio['Date']
factors_2['Date'] = factors['Date']



all_data = pd.merge(pf_2, factors_2, how='inner', on=['Date'])
all_data=sm.add_constant(all_data,prepend=True)

factors = ['const','4. close', 'squared', 'root', 'nasdaq_close', 'squared_nq' , 'root_nq', 'Value', 'squared_oil', 'root_oil', '1 YR', 'squared_tr', 'root_tr']
symbols = ['FLWS', 'FCTY', 'FCCY', 'SRCE', 'VNET', 'TWOU',
       'DGLD', 'JOBS', 'EGHT', 'AVHI', 'SHLM', 'AAON', 'ABAX', 'XLRN',
       'ACTA', 'BIRT', 'MULT', 'YPRO', 'AEGR', 'MDRX', 'EPAX', 'DOX',
       'UHAL', 'MTGE', 'CRMT', 'FOLD', 'BCOM', 'BOSC', 'HAWK', 'CFFI',
       'CHRW', 'KOOL', 'HOTR', 'PLCE', 'JRJC', 'CHOP', 'HGSH', 'HTHT',
       'IMOS', 'DAEG', 'DJCO', 'SATS', 'WATT', 'INBK', 'FTLB', 'QABA', 'GOOG']

weights = []
def run_regression(factors, symbols, dataframe):
    for symbol in symbols:
        if dataframe[symbol].isnull().sum(axis=0) <= 420:
            dataframe[symbol].fillna(method='bfill', inplace = True)
            dataframe[symbol].fillna(method='ffill', inplace = True)
            
            X = dataframe[factors].values
            X1= np.where(np.isnan(X), 0, X)
            y = dataframe[symbol].values
            y1 = np.where(np.isnan(y), 0,y)
            reg = LinearRegression()
            reg.fit(X1, y1)
            weights.append(reg.coef_)
        
        else:
           df_temp = all_data[np.isfinite(all_data[symbol])]
           X = df_temp[factors].values
           X1= np.where(np.isnan(X), 0, X)
           y = df_temp[symbol].values
           y1 = np.where(np.isnan(y), 0,y)
           reg = LinearRegression()
           reg.fit(X1, y1)
           weights.append(reg.coef_)
           

run_regression(factors,symbols, all_data)




factors_new=factors_2[['4. close', 'nasdaq_close', 'Value', '1 YR']]
factors_new.fillna(method='bfill', inplace = True)
factors_new.fillna(method='ffill', inplace = True)

factorCov=factors_new.cov()

factor1Mean=sum(factors_new['4. close'])/len(factors_new['4. close'])
factor2Mean=sum(factors_new['nasdaq_close'])/len(factors_new['nasdaq_close'])
factor3Mean=sum(factors_new['Value'])/len(factors_new['Value'])
factor4Mean=sum(factors_new['1 YR'])/len(factors_new['1 YR'])
factorMeans = [factor1Mean,factor2Mean,factor3Mean,factor4Mean]


sample = np.random.multivariate_normal(factorMeans, factorCov)

def fivePercentVaR(trials):
    numTrials = trials.count()
    topLosses = trials.takeOrdered(max(round(numTrials/20.0), 1))
    return topLosses[-1]


def sign(number):
    if number<0:
        return -1
    else:
        return 1

def featurize(factorReturns):
    factorReturns = list(factorReturns)
    squaredReturns = [sign(element)*(element)**2 for element in factorReturns]
    squareRootedReturns = [sign(element)*abs(element)**0.5 for element in factorReturns]
    # concat new features
    return squaredReturns + squareRootedReturns + factorReturns




def simulateTrialReturns(numTrials, factorMeans, factorCov, weights):
    trialReturns = []
    for i in range(0, numTrials):
        # generate sample of factors' returns
        trialFactorReturns = np.random.multivariate_normal(factorMeans, factorCov)
        
        # featurize the factors' returns
        trialFeatures = featurize(trialFactorReturns)
        
        # insert weight for intercept term
        trialFeatures.insert(0,1)
        
        trialTotalReturn = 0
        
        # calculate the return of each instrument
        # then calulate the total of return for this trial features
        for stockWeights in weights:
            instrumentReturn = sum([stockWeights[i] * trialFeatures[i] for i in range(len(trialFeatures))])
            trialTotalReturn += instrumentReturn
        
        trialReturns.append(trialTotalReturn)
    return trialReturns


simulateTrialReturns(1000,factorMeans,factorCov, weights)



if _name_ == "_main_":
#    if len(sys.argv) != 3:
#        print("Usage: pageRank <urlFile> <iterations>", file=sys.stderr)
#        sys.exit(-1)
        
        
    
    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    sc = spark.sparkContext
    
    parallelism = 12
    numTrials = 10000
    trial_indexes = list(range(0, parallelism))

    seedRDD = sc.parallelize(trial_indexes, parallelism)
    bFactorWeights = sc.broadcast(weights)

    trials = seedRDD.flatMap(lambda idx: \
                simulateTrialReturns(
                    max(int(numTrials/parallelism), 1), 
                    factorMeans, factorCov,
                    bFactorWeights.value
                ))
    trials.cache()


    valueAtRisk = fivePercentVaR(trials)

    print ("Value at Risk(VaR) 5%:", valueAtRisk)



spark.stop()
    

