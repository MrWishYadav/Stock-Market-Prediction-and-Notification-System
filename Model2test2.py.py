# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:33:12 2020

@author: visha
"""

#MODEL2TEST2.PY PROGRAM WHICH IS MODIFIED FOR MODEL2TEST.PY PROGRAM!
#This program is modified as per our needs but it is not able to diplay the candlesticks

#Import the libraries
#import math
#import pandas_datareader as web
#import numpy as np
import pandas as pd
#from nsepy import get_history
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM

# this library is used for the display of the graph for the closing price
#import matplotlib.pyplot as plt 

# Import the library you need to create the chart
import plotly.graph_objs as go

import datetime

# This plt function is used for data visualization of the Closing price v/s Date   
#plt.style.use('fivethirtyeight')

# this variable is initalized to obtain the date and time for predicting the closing price for tomorrow's data
NSECompanyHistoricDataToDate = datetime.datetime.now() + datetime.timedelta(days = 1)

#Getting and reading the stock quote from the NSEHistoricData file 
df=pd.read_csv("C:\\Users\\visha\\Desktop\\SM\\NSEHistoricData\\RELIANCE.CSV")

# data frame of the total number of (rows, columns) in which the data is appended
df.head()


data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])]

figSignal = go.Figure(data=data)
figSignal.show()


# Create a basic layout that names the chart and each axis.
layout = dict(
        title="RELIANCE_MODEL",
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title( text="Time")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title( text="Price in INR"))
)


# set the data from our data frame
data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])]

# display the Candlestic chart with the optional layout
figSignal = go.Figure(data=data, layout=layout)
figSignal.show()


from datetime import datetime
low_time = datetime(2019, 8, 22, 10, 33, 0, 0)
low_px = 179.91

high_time = datetime(2019, 8, 22, 9, 31, 0, 0)
high_px = 184.15

#######################################################
# Adding annotations for the high and low of the day.
annotations = []
annotations.append(go.layout.Annotation(x=low_time,
                                        y=low_px,
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowcolor="purple",
                                        arrowsize=2,
                                        arrowwidth=2,
                                        text="Low"))

annotations.append(go.layout.Annotation(x=high_time,
                                        y=high_px,
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowcolor="purple",
                                        arrowsize=2,
                                        arrowwidth=2,
                                        text="High"))


# Create a basic layout that names the chart and each axis.
layout = dict(
        title="RELIANCE",
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title( text="Time (EST - New York)"), rangeslider=dict (visible = False)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title( text="Price in INR")),
        width=1000,
        height=800,
        annotations=annotations
)

# set the data from our data frame
data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])]

# display the Candlestic chart with the optional layout
figSignal = go.Figure(data=data,layout=layout)
figSignal.show()


