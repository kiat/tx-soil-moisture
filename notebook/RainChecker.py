#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Libraries required
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

get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = [8, 6]
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# In[8]:


# Read and set the index to be the date

df_soil = pd.read_csv('../datasets/TX-Data/soil_station/SM_1.dat', sep=",", parse_dates=["Date"], index_col="Date")

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


df_soil


# In[5]:


ppt=df_soil['Ppt']
ppt[ppt!=0]
#Updates hourly!


# In[6]:


# Docs here https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot
get_ipython().run_line_magic('matplotlib', 'inline')
years = [str(item) for item in range(2015, 2022)]

for i in years:
    df_soil[i].SWC_5.plot(figsize=(12,6))
    plt.show()


# In[9]:


def findPeaks(SWC5):
    m=False
    peaks=[]
    index=[]
    cooldown=48
    rainpoint=.04
    forward_window=48
    backWindow=-12
    '''
    peaks- values of where we say it rains
    index- index at which it rains
    cooldown- minimum number of points between one rain point and another
    rainPoint- threshold difference to constitue rain
    forwardWindow- how far forward window checks to make sure slope does not continue to increase
    backWindow- how far back it checks the differences in rain values
    '''
    for i in range(24,len(SWC5)-48):
        c=SWC5[i]
        
        
        go=True
        for l in range(1,forward_window):
            if c<SWC5[i+l]:
                go=False
            if go:
                for j in range(backWindow,0):
                    if c-SWC5[i+j]>rainpoint:
                        m=True
                        peaks.append(SWC5[i])
                        index.append(i)
                        break
    for i in range(len(peaks)-1,0,-1):
        if index[i]-index[i-1]<cooldown:
            if peaks[i]>peaks[i-1]:
                peaks.pop(i-1)
                index.pop(i-1)
            else:
                peaks.pop(i)
                index.pop(i)
    return index, peaks
                    


# In[10]:


import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
years = [str(item) for item in range(2015, 2022)]

for i in years:
    q,a=findPeaks(df_soil[i].SWC_5)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,len(df_soil)+1)),y=df_soil[i].SWC_5, mode='lines', name='raw Data'))
    fig.add_trace(go.Scatter(x=q,y=a, mode='markers', name='rainpoints'))
    fig.show()


# As we can see, the algorithim for the most part works- unsure if what is requested is at the excact highpoint (would have been raining for a couple of hours there) or just find when it is leading to a high point- any advice?

# In[92]:





# In[ ]:




