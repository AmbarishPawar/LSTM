import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras_self_attention import SeqSelfAttention
from LSTM_functions import Format_data_for_LSTM,show_features_1D, get_layer_outputs,show_features_2D
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

train_data = loadmat('Gestures_for_LSTM_train.mat');
[FR_train,labels_train] = Format_data_for_LSTM(train_data)

F = FR_train[0]
#plt.pcolormesh(np.transpose(F),cmap='jet',vmin=0.3,vmax=1.3)
#plt.show()

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

#%% Evaluate the model on an unseen dataset from another day
test_data  = loadmat('Gestures_for_LSTM_test_0604.mat')
[FR_test,labels_test]   = Format_data_for_LSTM(test_data)
score = model.evaluate(FR_test,labels_test,verbose=0,batch_size=16)
#k = model.predict(FR_test)
print('Test loss:',score[0])
print('Test accuracy:',score[1])