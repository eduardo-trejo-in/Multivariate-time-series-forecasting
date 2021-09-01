!pip install yfinance

import yfinance as yf
from datetime import datetime, timedelta
import numpy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from tensorflow import keras
#from datetime import datetime

#_-----------------------------------------------

#Get the yesterday date
yesterday=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
#print("Fecha de ayer:", yesterday)
#get the stock quote

df= yf.download('TWTR',start='2013-11-18',end=yesterday)

df_topop=yf.download('TWTR',start='2013-11-18',end=yesterday)
#df= yf.download('TWTR',start='2013-11-18',end='2021-01-01')
#Get the number of rows and colums in the data set
#print(df["Close"].head())
#print("End")
print(df.head()) #7 columns, including the Date. 
print(type(df))

#_-----------------------------------------------

#Separate dates for future plotting
train_dates = df.index
train_date=pd.to_datetime(train_dates)
print(train_dates.shape)

#_-----------------------------------------------

#Variables for training
cols = list(df)[0:6]
#Date and volume columns are not used in training. 
print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']

#_-----------------------------------------------

#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)

print(len(df_for_training))


#_-----------------------------------------------

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
#Scaling for input X colums
scaler = StandardScaler()

scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

#Scaling for OutPuts "Y" colums
df_for_training_Open_scaled=df_for_training_scaled[:,[0]] #selecting colums from a numpy array
df_for_training_High_scaled=df_for_training_scaled[:,[1]]
df_for_training_Low_scaled=df_for_training_scaled[:,[2]]
df_for_training_Close_scaled=df_for_training_scaled[:,[3]]
df_for_training_Adj_Close_scaled=df_for_training_scaled[:,[4]]
df_for_training_Volume_scaled=df_for_training_scaled[:,[5]]

#_-----------------------------------------------
#Empty lists to be populated using formatted training data
trainX = []
trainY_Open = []
trainY_High = []
trainY_Low = []
trainY_Close = []
trainY_Adj_Close = []
trainY_Volume= []

n_future = 1   # Number of days we want to look into the future based on the past days.
n_past = 90  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).


for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    #trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
    #trainY_Open.append(df_for_training_scaled[i:i+1, 0:df_for_training.shape[1]])
    trainY_Open.append(df_for_training_Open_scaled[i:i+1, 0:df_for_training.shape[1]])
    trainY_High.append(df_for_training_High_scaled[i:i+1, 0:df_for_training.shape[1]])
    trainY_Low.append(df_for_training_Low_scaled[i:i+1, 0:df_for_training.shape[1]])
    trainY_Close.append(df_for_training_Close_scaled[i:i+1, 0:df_for_training.shape[1]])
    trainY_Adj_Close.append(df_for_training_Adj_Close_scaled[i:i+1, 0:df_for_training.shape[1]])
    trainY_Volume.append(df_for_training_Volume_scaled[i:i+1, 0:df_for_training.shape[1]])
 
trainX, trainY_Open, trainY_High, trainY_Low =  np.array(trainX), np.array(trainY_Open), np.array(trainY_High), np.array(trainY_Low)
trainY_Close, trainY_Adj_Close, trainY_Volume = np.array(trainY_Close), np.array(trainY_Adj_Close), np.array(trainY_Volume)

trainX=trainX[:len(trainX)-3]
trainY_Open=trainY_Open[:len(trainY_Open)-3]
trainY_High=trainY_High[:len(trainY_High)-3]
trainY_Low=trainY_Low[:len(trainY_Low)-3]
trainY_Close=trainY_Close[:len(trainY_Close)-3]
trainY_Adj_Close=trainY_Adj_Close[:len(trainY_Adj_Close)-3]
trainY_Volume=trainY_Volume[:len(trainY_Volume)-3]

print('trainX shape == {}.'.format(trainX.shape))
print('trainY_Open shape == {}.'.format(trainY_Open.shape))
print('trainY_High shape == {}.'.format(trainY_High.shape))
print('trainY_Low shape == {}.'.format(trainY_Low.shape))
print('trainY_Close shape == {}.'.format(trainY_Close.shape))
print('trainY_Adj_Close shape == {}.'.format(trainY_Adj_Close.shape))
print('trainY_Volume shape == {}.'.format(trainY_Volume.shape))
    

#_-----------------------------------------------

#-------------------- Define functional model simplified. ----------------------------------

keras.backend.clear_session()  # Reseteo sencillo
#we were here 8/25/2021 need to be inproved the model 
#Choose 
#https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
"""VIdeo Multiple outputs min 6:30
https://www.youtube.com/watch?v=JN08CqZKKkA&ab_channel=PythonEngineer"""

#https://www.tensorflow.org/guide/keras/functional

#---------Layes are created

inputs=keras.Input(shape=(trainX.shape[1:]))

x=LSTM_Layer1=keras.layers.LSTM(90, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)(inputs)
#x=Dropout_layer1=keras.layers.Dropout(0.2)(x)

"""x=LSTM_Layer2=keras.layers.LSTM(96, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)(x)
x=Dropout_layer2=keras.layers.Dropout(0.2)(x)

x=LSTM_Layer3=keras.layers.LSTM(96, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)(x)
x=Dropout_layer3=keras.layers.Dropout(0.2)(x)"""

#------------------missing to continue with symplification----------------------

x=LSTM_Layer4=keras.layers.LSTM(90, return_sequences=False)(x)
dense=Dropout_layer4=keras.layers.Dropout(0.4)(x)

#dense=keras.layers.Dense(80,activation='relu')(x)


#---------------------------Outputs
dense2=keras.layers.Dense(1,activation='relu')(dense)
dense2_2=keras.layers.Dense(1, activation='relu')(dense)
dense2_3=keras.layers.Dense(1, activation='relu')(dense)
dense2_4=keras.layers.Dense(1, activation='relu')(dense)
dense2_5=keras.layers.Dense(1, activation='relu')(dense)
dense2_6=keras.layers.Dense(1, activation='relu')(dense)

#-------Layers outputs are linked

outputs=dense2
outputs2=dense2_2
outputs3=dense2_3
outputs4=dense2_4
outputs5=dense2_5
outputs6=dense2_6

#-----The model it's created

model=keras.Model(inputs=inputs, outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Prediction')
#model=keras.Model(inputs=[inputs,None], outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Prediction')

print(model.summary())

keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

#_-----------------------------------------------

#------------------- Loss and optimizer ----------------------------------------
#got to ensure MeanAbsoluteError it's the good one for our data
loss1 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss2 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss3 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss4 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss5 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss6 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
optim=keras.optimizers.Adam(1e-3)
metrics=["accuracy"]

losses={
    "dense": loss1,
    "dense_1": loss2,
    "dense_2": loss3,
    "dense_3": loss4,
    "dense_4": loss5,
    "dense_5": loss6,
}

model.compile(loss=losses, optimizer=optim, metrics=metrics)

print(model.summary())

#--------------------------- Assing Y data to losses dictionary -----
y_data={ 
    "dense": trainY_Open,
    "dense_1": trainY_High,
    "dense_2": trainY_Low,
    "dense_3": trainY_Close,
    "dense_4": trainY_Adj_Close,
    "dense_5": trainY_Volume,
}

y_dataTrain={
    "dense": trainY_Open_For_Test,
    "dense_1": trainY_High_For_Test,
    "dense_2": trainY_Low_For_Test,
    "dense_3": trainY_Close_For_Test,
    "dense_4": trainY_Adj_Close_For_Test,
    "dense_5": trainY_Volume_For_Test,
}

#----------------------------- Training model -----------------
history = model.fit(trainX,y=y_data, epochs=50, batch_size=16)
#history = model.fit(trainX,y=y_data, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

test_scores = model.evaluate(trainX_For_Test, y_dataTrain, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['dense_1_loss'], label='Validation loss')
plt.plot(history.history['dense_1_accuracy'], label='dense_1_accuracy')
plt.legend()

#--------------------------------- -----------------
#Start with the last day in training date and predict future...
N_Days_to_predict=25#Redefining n_future to extend prediction dates beyond original n_future dates...

predict_period_dates = pd.date_range(list(train_dates)[-1], periods=N_Days_to_predict, freq='1d').tolist()
#predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_days_for_prediction, freq=us_bd).tolist()
print(predict_period_dates)

#--------------------------------- -----------------

#-------------------------------Forcasting-----------------------------...


Prediction_Saved=[]
Batch_to_predict=trainX[len(trainX)-6:len(trainX)-5]
#print(Batch_to_predict)
#print(Batch_to_predict.shape)
#print("--------------------------------")
for i in range(N_Days_to_predict):
  prediction = model.predict(Batch_to_predict) #the input is a 30 days batch
  prediction_Reshaped=np.reshape(prediction,(1,1,6))
  Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
  Batch_to_predict=np.delete(Batch_to_predict,0,1)
  print(Batch_to_predict.shape)
  #print(Batch_to_predict)
  #Batch_to_predict=prediction_Reshaped
  Prediction_Saved.append(prediction_Reshaped)
  #Perform inverse transformation to rescale back to original range
  #Since we used 5 variables for transform, the inverse expects same dimensions
  #Therefore, let us copy our values 5 times and discard them after inverse transform
#print(Batch_to_predict)
#print(Batch_to_predict)
"""print(len(Prediction_Saved))#<----------- 20 days predicted
print(Prediction_Saved)"""

y_pred_future = scaler.inverse_transform(Prediction_Saved)[:,0]
y_pred_future=y_pred_future
print(y_pred_future[:3])
predict_Open=[]
predict_High=[]
predict_Low=[]
predict_Close=[]
predict_Adj_Close=[]
predict_Volume=[]

for i in range(len(y_pred_future)):
  predict_Open.append(y_pred_future[i][0][0])
  predict_High.append(y_pred_future[i][0][1])
  predict_Low.append(y_pred_future[i][0][2])
  predict_Close.append(y_pred_future[i][0][3])
  predict_Adj_Close.append(y_pred_future[i][0][4])
  predict_Volume.append(y_pred_future[i][0][5])

print(len(predict_Open))

#-------------------------------Forcasting-----------------------------...
# Convert timestamp to date
forecast_dates = []
print(len(predict_period_dates))
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
print(len(forecast_dates))
print("-----------------------")

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':predict_Open})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


original = df['Open']
#original['Date']=pd.to_datetime(original['Date'])
#original = original.loc[original['Date'] >= '2020-5-1']
print(original[len(original)-90:])
print(df_forecast)

plt.plot(original[len(original)-90:])

plt.plot(df_forecast['Date'], df_forecast['Open'])
sns.lineplot(original[len(original)-90:])
sns.lineplot(df_forecast['Date'], df_forecast['Open'])
