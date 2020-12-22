# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:24:15 2020

@author: visha
"""

# Model2test3.py is a program which depicts the candlesticks of all the OHLC in a single graph
# Note- first install the package named 'pip install mpl_finance'
# The class and object should be added and the model mmust be added 
# The resource link -->>  https://saralgyaan.com/posts/python-candlestick-chart-matplotlib-tutorial-chapter-11/


# importing the basic libraries
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd
import matplotlib.dates as mpl_dates


# This plt function is used for data visualization of candlesticks of the RELIANCE_price in INR v/s Date 
plt.style.use('ggplot')

# Extracting Data for plotting
data = pd.read_csv('C:\\Users\\visha\\Desktop\\SM\\NSEHistoricData\\RELIANCE.csv')
ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

# Creating Subplots
fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.5, colorup='green', colordown='red', alpha=0.5)

# Setting labels & titles
ax.set_xlabel('<----------Date---------->')
ax.set_ylabel('<--------RELIANCE_Price in INR--------->')
fig.suptitle('RELIANCE_STOCK_GROWTH')

# Formatting Date
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()


