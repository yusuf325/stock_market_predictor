import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import keras


df = pd.read_csv(os.path.join('stock_market_predictor', 'Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
df = df.sort_values('Date')

plt.figure(figsize = (20, 10))
plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
plt.xticks(range(0, df.shape[0], 500),df['Date'].loc[::500],rotation = 45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

highest = df.loc[:, 'High'].to_numpy()
lowest = df.loc[:, 'Low'].to_numpy()
mid = (highest + lowest) / 2.0

train_mid = mid[:10000]
test_mid = mid[10000:]

train_mid = train_mid.reshape(-1, 1)

scaler = MinMaxScaler(feature_range = (0, 1))
scaled_train_mid = scaler.fit_transform(train_mid)






xtrain = []
ytrain = []
for i in range(60, len(scaled_train_mid)):
    xtrain.append(scaled_train_mid[i - 60: i, 0])
    ytrain.append(scaled_train_mid[i, 0])
    
xtrain, ytrain = np.array(xtrain), np.array(ytrain)

model = keras.models.Sequential()


first_lstm_layer = keras.layers.LSTM(units=200,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(xtrain.shape[1],1))
second_lstm_layer = keras.layers.LSTM(units=200,kernel_initializer='glorot_uniform',return_sequences=True)
third_lstm_layer = keras.layers.LSTM(units=150,kernel_initializer='glorot_uniform',return_sequences=True)
output_layer = keras.layers.Dense(units = 1)

model.add(first_lstm_layer)
model.add(keras.layers.Dropout(0.2))
model.add(second_lstm_layer)
model.add(keras.layers.Dropout(0.2))
model.add(third_lstm_layer)
model.add(keras.layers.Dropout(0.2))
model.add(output_layer)

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(xtrain, ytrain, batch_size = 500, epochs = 50)

total= pd.concat([train_mid['Open'], test_mid['Open']], axis=0) 
test_input = total[len(total) - len(test_mid) - 60:].values
test_input= test_input.reshape(-1, 1)
test_input= scaler.transform(test_input)

xtest= []
for i in range(60,80):
    xtest.append(test_input[i-60:i, 0])
    
xtest= np.array(xtest)

xtest= np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))
predicted_value= model.predict(xtest)

predicted_value= scaler.inverse_transform(predicted_value)

plt.figure(figsize=(20,10))
plt.plot(test_mid,'red',label='Real Prices')
plt.plot(predicted_value,'blue',label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Prices')
plt.title('Real vs Predicted Prices')
plt.legend(loc='best', fontsize=20)