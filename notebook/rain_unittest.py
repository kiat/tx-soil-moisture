import os
import datetime

import IPython
import IPython.display

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import fnmatch
import plotly.graph_objects as go
# %matplotlib inline
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

# Read and set the index to be the date

df_soil = pd.read_csv('/Users/shuhanshen/Desktop/Research/tx-soil-moisture-main- week1/datasets/TX-Data/soil_station/SM_1.dat', sep=",", parse_dates=["Date"], index_col="Date")

# pandas remove spaces in the column names
df_soil.columns = df_soil.columns.str.replace(' ','')
df_soil['SWC_5'] = df_soil['SWC_5'].astype(float)
df_soil['SWC_10'] = df_soil['SWC_10'].astype(float)
df_soil['SWC_20'] = df_soil['SWC_20'].astype(float)
df_soil['SWC_50'] = df_soil['SWC_50'].astype(float)

df_soil['T_5'] = df_soil['T_5'].astype(float)
df_soil['T_10'] = df_soil['T_10'].astype(float)
df_soil['T_20'] = df_soil['T_20'].astype(float)
df_soil['T_50'] = df_soil['T_50'].astype(float)

df_soil['Ppt'] = df_soil['Ppt'].astype(float)


ppt=df_soil['Ppt']
ppt[ppt!=0]
#Updates hourly!
# print(df_soil.SWC_5)
def findPeaks(SWC5, thresh, rainpoint):
    m = False
    peaks = []
    index = []
    cooldown = 24
    forward_window = 12
    backWindow = -12

    for i in range(24, len(SWC5) - forward_window):
        c = SWC5[i]
        # Loops through all hours within the year, except boundry hours

        go = True
        # If a greater value detected in the near future, not at a peak
        for l in range(1, forward_window):
            if c < SWC5[i + l]:
                go = False
            # If at a possible peak
            if go:
                for j in range(backWindow, 0):
                    # If signifigantly greater than some point in the past and value is greater than some threshold value, mark as a perak
                    if c - SWC5[i + j] > rainpoint and c > thresh:
                        m = True
                        peaks.append(SWC5[i])
                        index.append(i)
                        break
    # Gets rid of peaks that are within cooldown of each other- unmark the lower peak
    for i in range(len(peaks) - 1, 0, -1):
        if index[i] - index[i - 1] < cooldown:
            if peaks[i] > peaks[i - 1]:
                peaks.pop(i - 1)
                index.pop(i - 1)
            else:
                peaks.pop(i)
                index.pop(i)
    return index, peaks


def Compare(SWC5, ppt, t, t2, r, r2):


    q,a=findPeaks(SWC5, t, r)
    w,s=findPeaks(ppt, t2, r2)
    #The arrays below will be the corrosponding elements in q and 2 and subsequently a and s which match
    myrainIndex=np.full(len(w), False)
    myMoistureIndex=np.full(len(q), False)
    for rainIndex in range(len(w)):
        #Checks all rain indecies
        for i in range(len(q)):
            #Against all moisture points
            for t in range(-24,25):
                #If within a windoow of 48 hours, these twon points match!
                if w[rainIndex]==q[i]+t:
                    myrainIndex[rainIndex]=True
                    myMoistureIndex[i]=True
                    break
    q=np.array(q)
    a=np.array(a)
    w=np.array(w)
    s=np.array(s)
    percentMoisture=100*len(q[myMoistureIndex==True])/len(q)
    #Percentage of matching moisture and rain points, respectively
    percentRain=100*len(w[myrainIndex==True])/len(w)
    q = list(q)
    a = list(a)
    w = list(w)
    s = list(s)
    return q, a, w, s , percentMoisture, percentRain

def findIncreasing(SWC5):
    peaks = []
    index = []
    backWindow = -1

    for i in range(24, len(SWC5)):
        c = SWC5[i]
        record = False
        # Loops through all hours within the year, except boundry hours
        # If signifigantly greater than some point in the past and value is greater than some threshold value, mark as a perak
        for j in range(-12, -6):
            if c - SWC5[i + j] > 0.03:
                record = True
                break
        if record:
            peaks.append(SWC5[i])
            index.append(i)
    return index, peaks

def Compare2(SWC5, ppt):

    #Should I also check all non-zeros rainvalues with some increase in soil moisture?
    q,a=findIncreasing(SWC5)
    #The arrays below will be the corrosponding elements in q and 2 and subsequently a and s which match
    myMoistureIndex=np.full(len(q), False)
    for rainIndex in range(len(q)):
        #Checks all rain indecies
        for t in range(-24,1):
                #If within a windoow of 48 hours, these twon points match!
                if ppt[q[rainIndex]+t]!=0:
                    myMoistureIndex[rainIndex]=True
                    break
    q=np.array(q)
    a=np.array(a)
    percentMoisture=100*len(q[myMoistureIndex==True])/len(q)
    q=list(q)
    a=list(a)
    #Percentage of matching moisture and rain points, respectively
    return q, a, percentMoisture
