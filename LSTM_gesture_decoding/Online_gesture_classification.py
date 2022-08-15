"""
@author: ambarish

This script does live (online) classification of neural spiking data as it
arrives at the decoder. While the LSTM decoder is trained on a dataset of 
neatly aligned trials, in a practical scenario while the patient goes about 
their daily life, the decoder does not get neatly-aligned data. The real 
challenge for a decoder is to decode gestures as the patient performs them, in 
real time, with no prior information on which gesture will happen next. This 
script simulates a "real-life" scenario where neural data is streamed into the
decoder, and the LSTM continuously decodes data as it comes in. 
Here are the steps
- Load a pretrained LSTM model that was trained by the LSTM_gestures script.
- Load one block of neural spiking data (different from what was used to train decoder).
- Each block contains 14 gestures of hand in random order: rest, open, pinch
- 0 - Rest, 1 - Open, 2 - Pinch. This is the convention used.
- Data is (pseudo)streamed in once per time-step. This is done by gradually 
  filling up buffers. This is done to simulate a real BCI experiment.
- As the buffers start filling up, whatever data has streamed in is fed into 
  the LSTM for prediction into 1 of the three grasp states.
- In the end, the script plots the ground-truth data (i.e what gesture the
  patient was instructed to do), z-scored firing rate from one electrode, and
  the predicted gesture. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tensorflow import keras
from LSTM_functions_online import Process_and_classify_per_block

model = keras.models.load_model('LSTM_trained_model')
#%% Load data from block
data         = loadmat('online_data.mat')

#%%
decode_win  = 30 #how many samples to decode at one instance. The larger this
#number, better the decoding, but this can slow down the decoder. This number
#is a trade-off between speed and accuracy
threshold   = 0.8 #If the predictions within 'decode_win' exceed 80% threshold
#then the current prediction of the decoder will be that particular gesture. 
#Otherwise decoder defaults to 'rest' gesture. 

# This function does the heavy lifting of actually decoding gestures in real-time
# Detailed comments are present within the function
[ypred,ytest,real_timebin,gest_marks,whole_buffer,pred_smooth,block_start,
 block_end,t] = Process_and_classify_per_block(model,data,decode_win,threshold)

#%% Reshaping to match dimensions for plotting later
real_gest = np.reshape(np.transpose(gest_marks[:,1:]),t.shape)

#%% Plotting
which_elect = 23 #Get activity of electrode #23. Could do any electode here
elec_to_plot = whole_buffer[:,which_elect]
# Spiking data is noisy. Smooth here with a 50 sample window for illustration
elec_to_plot = np.convolve(elec_to_plot,np.ones(50)/50,'same')

# Ground-truth/cued gestures, what the patient was instructed and when
plt.subplot(3,1,1)
plt.plot(t,np.float64(real_gest),'r-'),plt.xlim([block_start,block_end])

# Z-scored firing rate from 'which_elect' above
plt.subplot(3,1,2)
plt.plot(t,elec_to_plot),plt.xlim([block_start,block_end])

# Predicted gestures. 
plt.subplot(3,1,3)
plt.plot(t,pred_smooth),plt.xlim([block_start,block_end])
plt.xlabel('time(s)')