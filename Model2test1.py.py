
"""
Created on Wed May 6 12:30:01 2020

@author: Vishal Gajendrarao Yadav
"""
# This code is able to load the Reliance Dataset and it is able to train the Open, High, Low, Close(OHLC) of the National Stock Exchange(NSE)
# This code is also able to predict the OHLC based on the previous day's close price
# This bar graph depicts the overall history of the RELIANCE company along with individual parameters on x-y co-ordinates


#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from nsepy import get_history
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# this library is used for the display of the graph for the closing price
import matplotlib.pyplot as plt 

import datetime

# This plt function is used for data visualization of the Closing price v/s Date   
plt.style.use('fivethirtyeight')

# this variable is initalized to obtain the date and time for predicting the closing price for tomorrow's data
NSECompanyHistoricDataToDate = datetime.datetime.now() + datetime.timedelta(days = 1)

#Getting and reading the stock quote from the NSEHistoricData file 
df=pd.read_excel("C:\\Users\\visha\\Desktop\\SM\\NSEHistoricData\\RELIANCE.xlsx", sheet_name="RELIANCE")

# data frame of the total number of (rows, columns) in which the data is appended
df.shape

# Creating a Class and object for OHLC data values

#class TA:
#    def 

#Visualize the closing price history

# This displays the total size of the figure (16=rows, 8=columns)
#plt.figure(figsize=(16,8))

# The title of the plot graph before prediction 
#plt.title('Close Price History')

# The plotting of the closing price from the excel sheet (Column = 'I')
#plt.plot(df['Close'])

# The x-axis and y-axis labels with the fontsize of 18
#plt.xlabel('Date',fontsize=18)
#plt.ylabel('Close Price INR',fontsize=18)

# Display of the actual closing price for the (predicted date -1 date)
#plt.show()


#Create a new dataframe with only the 'Close' column
data1 = df.filter(['Open'])
data2 = df.filter(['High'])
data3 = df.filter(['Low'])
data4 = df.filter(['Close'])

#Converting the dataframe to a numpy array
dataset1 = data1.values
dataset2 = data2.values
dataset3 = data3.values
dataset4 = data4.values


#Get /Compute the model with only 80% /(0.8) of the number of rows dataset
training_data_len1 = math.ceil( len(dataset1) *.8) 
training_data_len2 = math.ceil( len(dataset2) *.8)
training_data_len3 = math.ceil( len(dataset3) *.8)
training_data_len4 = math.ceil( len(dataset4) *.8)

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset1)
scaled_data = scaler.fit_transform(dataset2)
scaled_data = scaler.fit_transform(dataset3)
scaled_data = scaler.fit_transform(dataset4)

#Create the scaled training data set 
train_data1 = scaled_data[0:training_data_len1  , : ]
train_data2 = scaled_data[0:training_data_len2  , : ]
train_data3 = scaled_data[0:training_data_len3  , : ]
train_data4 = scaled_data[0:training_data_len4  , : ]


#Split the data into x_train and y_train data sets
a_train=[]
b_train = []
for i in range(60,len(train_data1)):
    a_train.append(train_data1[i-60:i,0])
    b_train.append(train_data1[i,0])
    
c_train=[]
d_train = []
for j in range(60,len(train_data2)):
    c_train.append(train_data2[j-60:j,0])
    d_train.append(train_data2[j,0])


e_train=[]
f_train = []
for k in range(60,len(train_data3)):
    e_train.append(train_data3[k-60:k,0])
    f_train.append(train_data3[k,0])


g_train=[]
h_train = []
for l in range(60,len(train_data4)):
    g_train.append(train_data4[l-60:l,0])
    h_train.append(train_data4[l,0])    

#Convert x_train and y_train to numpy arrays
a_train, b_train = np.array(a_train), np.array(b_train)
c_train, d_train = np.array(c_train), np.array(d_train)
e_train, f_train = np.array(e_train), np.array(f_train)
g_train, h_train = np.array(g_train), np.array(h_train)

#Reshape the data into the shape accepted by the LSTM
a_train = np.reshape(a_train, (a_train.shape[0],a_train.shape[1],1))
c_train = np.reshape(c_train, (c_train.shape[0],c_train.shape[1],1))
e_train = np.reshape(e_train, (e_train.shape[0],e_train.shape[1],1))
g_train = np.reshape(g_train, (g_train.shape[0],g_train.shape[1],1))

#Build the LSTM network model
model1 = Sequential()
model1.add(LSTM(units=50, return_sequences=True,input_shape=(a_train.shape[1],1)))
model1.add(LSTM(units=50, return_sequences=False))
model1.add(Dense(units=25))
model1.add(Dense(units=1))

model2 = Sequential()
model2.add(LSTM(units=50, return_sequences=True,input_shape=(c_train.shape[1],1)))
model2.add(LSTM(units=50, return_sequences=False))
model2.add(Dense(units=25))
model2.add(Dense(units=1))

model3 = Sequential()
model3.add(LSTM(units=50, return_sequences=True,input_shape=(e_train.shape[1],1)))
model3.add(LSTM(units=50, return_sequences=False))
model3.add(Dense(units=25))
model3.add(Dense(units=1))

model4 = Sequential()
model4.add(LSTM(units=50, return_sequences=True,input_shape=(g_train.shape[1],1)))
model4.add(LSTM(units=50, return_sequences=False))
model4.add(Dense(units=25))
model4.add(Dense(units=1))

#Compile the model
model1.compile(optimizer='adam', loss='mean_squared_error')
model2.compile(optimizer='adam', loss='mean_squared_error')
model3.compile(optimizer='adam', loss='mean_squared_error')
model4.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model1.fit(a_train, b_train, batch_size=1, epochs=1)
model2.fit(c_train, d_train, batch_size=1, epochs=1)
model3.fit(e_train, f_train, batch_size=1, epochs=1)
model4.fit(g_train, h_train, batch_size=1, epochs=1)

#Test data set
test_data1 = scaled_data[training_data_len1 - 60: , : ]
test_data2 = scaled_data[training_data_len2 - 60: , : ]
test_data3 = scaled_data[training_data_len3 - 60: , : ]
test_data4 = scaled_data[training_data_len4 - 60: , : ]

#Create the x_test and y_test data sets
a_test = []
b_test =  dataset1[training_data_len1 : , : ] 
#Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data1)):
    a_test.append(test_data1[i-60:i,0])
    
    
c_test = []
d_test =  dataset2[training_data_len2 : , : ] 
#Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for j in range(60,len(test_data2)):
    c_test.append(test_data2[j-60:j,0])


e_test = []
f_test =  dataset3[training_data_len3 : , : ] 
#Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for k in range(60,len(test_data3)):
    e_test.append(test_data3[k-60:k,0])


g_test = []
h_test =  dataset4[training_data_len4 : , : ] 
#Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for l in range(60,len(test_data4)):
    g_test.append(test_data4[l-60:l,0])    
    
#Convert x_test to a numpy array 
a_test = np.array(a_test)
c_test = np.array(c_test)
e_test = np.array(e_test)
g_test = np.array(g_test)

#Reshape the data into the shape accepted by the LSTM
a_test = np.reshape(a_test, (a_test.shape[0],a_test.shape[1],1))
c_test = np.reshape(c_test, (c_test.shape[0],c_test.shape[1],1))
e_test = np.reshape(e_test, (e_test.shape[0],e_test.shape[1],1))
g_test = np.reshape(g_test, (g_test.shape[0],g_test.shape[1],1))

#Getting the models predicted price values
predictions1 = model1.predict(a_test) 
predictions1 = scaler.inverse_transform(predictions1)#Undo scaling

predictions2 = model2.predict(c_test) 
predictions2 = scaler.inverse_transform(predictions2)

predictions3 = model3.predict(e_test) 
predictions3 = scaler.inverse_transform(predictions3)

predictions4 = model4.predict(e_test) 
predictions4 = scaler.inverse_transform(predictions4)

#Calculate/Get the value of RMSE
rmse1=np.sqrt(np.mean(((predictions1- b_test)**2)))
print(rmse1)

rmse2=np.sqrt(np.mean(((predictions2- d_test)**2)))
print(rmse2)

rmse3=np.sqrt(np.mean(((predictions3- f_test)**2)))
print(rmse3)

rmse4=np.sqrt(np.mean(((predictions4- h_test)**2)))
print(rmse4)

#Plot/Create the data for the graph
train1 = data1[:training_data_len1]
valid1 = data1[training_data_len1:]
valid1['Predictions1'] = predictions1


train2 = data2[:training_data_len2]
valid2 = data2[training_data_len2:]
valid2['Predictions2'] = predictions2


train3 = data3[:training_data_len3]
valid3 = data3[training_data_len3:]
valid3['Predictions3'] = predictions3


train4 = data4[:training_data_len4]
valid4 = data4[training_data_len4:]
valid4['Predictions4'] = predictions4


#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=20)
plt.ylabel('Open Price INR', fontsize=20)
plt.plot(train1['Open'])
plt.plot(valid1[['Open', 'Predictions1']])
plt.legend(['Train', 'Val', 'Predictions'], loc='Closeer right')
plt.show()


plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=20)
plt.ylabel('High Price INR', fontsize=20)
plt.plot(train2['High'])
plt.plot(valid2[['High', 'Predictions2']])
plt.legend(['Train', 'Val', 'Predictions'], loc='Closeer right')
plt.show()


plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=20)
plt.ylabel('Low Price INR', fontsize=20)
plt.plot(train3['Low'])
plt.plot(valid3[['Low', 'Predictions3']])
plt.legend(['Train', 'Val', 'Predictions'], loc='Closeer right')
plt.show()


plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=20)
plt.ylabel('Close Price INR', fontsize=20)
plt.plot(train4['Close'])
plt.plot(valid4[['Close', 'Predictions4']])
plt.legend(['Train', 'Val', 'Predictions'], loc='Closeer right')
plt.show()

#Show the valid and predicted prices
print(valid1)
print(valid2)
print(valid3)
print(valid4)

#Get the quote
stock_quote=pd.read_excel("C:\\Users\\visha\\Desktop\\SM\\NSEHistoricData\\RELIANCE.xlsx", sheet_name="RELIANCE")
#Create a new dataframe
new_df1 = stock_quote.filter(['Open'])

new_df2 = stock_quote.filter(['High'])

new_df3 = stock_quote.filter(['Low'])

new_df4 = stock_quote.filter(['Close'])

#Get the last 60 day closing price 
last_60_days1 = new_df1[-60:].values

last_60_days2 = new_df2[-60:].values

last_60_days3 = new_df3[-60:].values

last_60_days4 = new_df4[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled1 = scaler.transform(last_60_days1)

last_60_days_scaled2 = scaler.transform(last_60_days2)

last_60_days_scaled3 = scaler.transform(last_60_days3)

last_60_days_scaled4 = scaler.transform(last_60_days4)

#Create an empty list
A_test = []

C_test = []

E_test = []

G_test = []


#Append teh past 60 days
A_test.append(last_60_days_scaled1)

C_test.append(last_60_days_scaled2)

E_test.append(last_60_days_scaled3)

G_test.append(last_60_days_scaled4)

#Convert the X_test data set to a numpy array
A_test = np.array(A_test)

C_test = np.array(C_test)

E_test = np.array(E_test)

G_test = np.array(G_test)

#Reshape the data
A_test = np.reshape(A_test, (A_test.shape[0], A_test.shape[1], 1))

C_test = np.reshape(C_test, (C_test.shape[0], C_test.shape[1], 1))

E_test = np.reshape(E_test, (E_test.shape[0], E_test.shape[1], 1))

G_test = np.reshape(G_test, (G_test.shape[0], G_test.shape[1], 1))

#Get the predicted scaled price
pred_price1 = model1.predict(A_test)

pred_price2 = model2.predict(C_test)

pred_price3 = model3.predict(E_test)

pred_price4 = model4.predict(G_test)
#undo the scaling 
pred_price1 = scaler.inverse_transform(pred_price1)
print(pred_price1)

pred_price2 = scaler.inverse_transform(pred_price2)
print(pred_price2)

pred_price3 = scaler.inverse_transform(pred_price3)
print(pred_price3)

pred_price4 = scaler.inverse_transform(pred_price4)
print(pred_price4)

