import pandas_datareader as data
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

start = '2015-1-01'
end = '2022-10-31'

st.title('Stock Trend Pridiction')

user_input = st.text_input('Enter CRYPTO Coin', 'BTC')
user_input = user_input + '-USD'
df = data.DataReader(user_input, 'yahoo', start, end)

# describing data
st.subheader("Data from 2015 to 2022")
st.write(df.describe())

#visualizations
st.subheader("Closing Price Vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price Vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader("Closing Price Vs Time Chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

# splittling data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

# print(data_training.shape)
# print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


# load my model
model = load_model('keras_model.h5')



#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing])
input_data = scaler.fit_transform(final_df)

X_test = []
Y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i - 100: i])
    Y_test.append(input_data[i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test)
Y_predicted = model.predict(X_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
Y_predicted = Y_predicted * scale_factor
Y_test = Y_test * scale_factor


# final graph
st.subheader('Predictions Vs. Originals')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(Y_test, 'b', label = 'Original Price', linewidth=2)
plt.plot(Y_predicted, 'r', label = 'predicted Price', linewidth=2)
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

