"""
author @ambarish

This is a simmple script that trains a Long Short-Term Memory (LSTM) network on 
trial-aligned neural data. 

Neural data is a series of discrete events indicated by the timestamps on when 
'spikes' occurred. Whenever a spike occurs, the timestamp when that spike occurred
is stored. This discrete event is further processed into 'spike-rate', by chosing
a time window, within which the number of spikes that occurred is converted into 
a rate. This spike rate is used to train the LSTM. Here, we have spike rates from
128 channels, or electrodes that are implanted in the motor and sensory cortices.

In the experiment, the patient is cued to perform the following gestures of the left hand
0 - Rest. Hand is kept in a resting position, the default position
1 - Open. Open/splay the fingers as wide as possible
2 - Pinch. Thumb and forefinger pinch gesture

Each dataset contains 9 blocks of data, each containing 14 gestures randomly
interspersed, for a total of 126 repetitions. The data has been preprocessed 
to be square-rooted, z-scored, and smoothed. In the scipt, data is further 
sorted by electrodes and repetitions and aligned according to cue onset and offset
of gestures. Thus, we only feed ths gesture part of the timestamps into the LSTM. 
Most of the processing is done in the 'Format_data_for_LSTM' function.

There are total of 6 datasets. We train on one dataset, and evaluate the performance
of the LSTM on the other 5 unseen datasets. 
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from LSTM_functions import Format_data_for_LSTM,show_features_1D, get_layer_outputs,show_features_2D
from scipy.io import loadmat

# Load preprocessed training data
train_data = loadmat('Gestures_for_LSTM_train.mat');

# Align data according to trial start and stop times and electrodes
[FR_train,labels_train] = Format_data_for_LSTM(train_data)

# Colormap plots to check that processing is correct (uncomment below)
# F = FR_train[0]
# plt.pcolormesh(np.transpose(F),cmap='jet',vmin=0.3,vmax=1.3)
# plt.show()

# Create a two layer LSTM model, with variable time input
model = Sequential()
model.add(LSTM(128, return_sequences=True,input_shape=(None,FR_train.shape[2]),
               activation = 'tanh',name='lstm1'))
model.add(Dropout(0.4))
model.add(LSTM(128, return_sequences=False,activation = 'tanh',
               name='lstm2'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax',name='dense1'))
model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
model.fit(FR_train,labels_train, epochs=10, batch_size=16)

print(model.summary())

#%% 
# Evaluate the model on an unseen dataset from another day
# There are 5 datasets the model can be evaluated on
test_data  = loadmat('Gestures_for_LSTM_test_0604.mat')
[FR_test,labels_test]   = Format_data_for_LSTM(test_data)
score = model.evaluate(FR_test,labels_test,verbose=0,batch_size=16)
print('Test loss:',score[0])
print('Test accuracy:',score[1])