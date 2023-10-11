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



window_size = 100
N = train_mid.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    
    date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_mid[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_mid[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),mid,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()



window_size = 100
N = train_mid.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_mid[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_mid[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),mid,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()





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