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
- Load one block of neural spiking data recorded from a different day.
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

model = keras.models.load_model('LSTM_model_sm')
#%% Load data from block
data         = loadmat('online_data.mat')

#%% Process data as if it is streaming from the brain
fr           = data['fr']           #Firing rate
gest_marks   = data['gest_marks']   #Which gesture in real-time
real_timebin = data['real_timebin'] #Actual timeline in seconds
cue_on_nev   = data['cue_on_nev']   #Timestamp of when gesture started
cue_off_nev  = data['cue_off_nev']  #Timestamp of when gesture ended

decode_win  = 30 #how many samples to decode at one instance. The larger this
#number, better the decoding, but this can slow down the decoder. This number
#is a trade-off between speed and accuracy

# Initialize buffers and other variables
z_buffer    = np.zeros((2000,128))
time_buffer = np.zeros((1,2000))
fr_buffer   = np.full([2000,128],np.nan) #Create array of nans
zscore_part = np.zeros((1,decode_win,128))

used_nsp     = 2 #Neural data from right-hemisphere, left hand gestures
fr_nsp       = fr[used_nsp,:,:]
whole_buffer = np.zeros((fr_nsp.shape[1],128))
fr_sqrt      = np.zeros((fr_nsp.shape[0],))
YPred        = np.zeros((fr_nsp.shape[1],3))
YClass       = np.zeros((fr_nsp.shape[1],))
pred_smooth  = np.zeros((fr_nsp.shape[1],)) 
bins         = [0,1,2,3]
threshold    = 0.8

for counter_packets in range(0,fr_nsp.shape[1]):
    print(counter_packets)
    fr_sqrt     = np.sqrt(fr_nsp[:,counter_packets]) #square-root of firing rate
    fr_buffer   = np.roll(fr_buffer,-1,0) #shift rows, to accomodate new data
    time_buffer = np.roll(time_buffer,-1,1)
    
    fr_buffer[-1,:] = fr_sqrt
    time_buffer[-1] = real_timebin[0,counter_packets]
    fr_mn = np.nanmean(fr_buffer,0) #mean and std taken to calculate z-score
    fr_sd = np.nanstd(fr_buffer,0,ddof=1)
    
    # if buffer has only 1 column, std of one column will be zero, and z-score
    # calculation will yield NaNs. This loop gets over that condition.
    if counter_packets>0:
        z_score = ((fr_sqrt-fr_mn)/fr_sd)
        z_score[np.isnan(z_score)] = 0
        z_score[np.isinf(z_score)] = 0
        
        z_buffer = np.roll(z_buffer,-1,0)
        z_buffer[-1,:] = z_score
        
    else:
        # Do not divide by std, otherwise we'll get only NaNs
         z_buffer[-1,:] = (fr_sqrt-fr_mn)/2
    
    # whole_buffer stored all z-scored data, so that we can examine it later 
    # if we wanted to. 
    whole_buffer = np.roll(whole_buffer,-1,0)
    whole_buffer[-1,:] = z_buffer[-1,:]
    
    # zscore_part only takes 'decode_win' number of samples that will be fed to
    # the decoder. 
    zscore_part[0,:,:] = z_buffer[-decode_win:,:]
    t_part = time_buffer[-decode_win:,:]
    # Predict the gesture based on 'decode_win' amount of z-scored data
    Y = model.predict(zscore_part)
    YPred[counter_packets,:] = Y
    YClass[counter_packets] = Y.argmax(axis=-1)# convert probabilities to 
    # integer states
    
    # This loop does a form of smoothing. The predicted gesture within a 
    # 'decode_win' is the one that exceeds a 'threshold' amount. For example:
    # if the 'pinch' gesture happens 25 times in a 30 sample window, 25 exceeds 
    # threshold*30. So then the prediction within this window is 'pinch'. If 
    # no gesture exceeds threshold*30, then the decoder defaults to 'rest'.
    if counter_packets > (decode_win-1):
        classify_in_window,b = np.histogram(YClass[counter_packets-decode_win:counter_packets],bins)    
        max_class = np.max(classify_in_window)
        max_class_idx = np.argmax(classify_in_window)
        if max_class>=(threshold*decode_win):
            pred_smooth[counter_packets] = max_class_idx
        else:
            pred_smooth[counter_packets] = 0 # default to rest gesture
    continue

# We want to find when the block actually starts. Get the times here. 
time_bins_block = [cue_on_nev[1]-0.5,cue_off_nev[-1]+3]
block_start_idx = np.abs(real_timebin-time_bins_block[0]).argmin();
block_end_idx   = np.abs(real_timebin-time_bins_block[-1]).argmin();

#%% Getting classification accuracy
reaction_delay = 0
decode_delay   = 0.35 
# start decoding after a 'total_delay'. This delay is basically the time to react
# for the patient after they get the cue to make a hand gesture. 
total_delay    = reaction_delay+decode_delay
decode_end     = total_delay+2.5 #stop decoding after 2.5s. This is when the 
# patient has stopped their gestures. This is an empirical estimate

# Reshaping to match dimensions for plotting later
t = real_timebin
t = np.reshape(t,t.shape[1],)

# Get the predictied gestures within a time-window defined by when the patient
# was cued to start and stop gesture. Ideally the 'ground-truth' of the time
# duration of cued gestures and should closely align with predicted gestures. 
# However, there are delays in when the participant makes gestures, delays in 
# the LSTM decoder. These need to be taken into account before calculating 
# accuracy.
YMov = np.zeros(len(cue_on_nev))
for c in range(len(cue_on_nev)):
    this_cue_on = cue_on_nev[c]
    this_cue_off = this_cue_on+decode_end
    cue_idx = np.where(np.logical_and(t>=(this_cue_on+total_delay),t<=this_cue_off))
    this_pred = pred_smooth[cue_idx]
    classified_as_what,temp1 = np.histogram(this_pred,[0,1,2,3])
    max_count_idx = np.argmax(classified_as_what)
    YMov[c] = max_count_idx

# Simple reshaping to match dimensions for plotting
t = t[1:]
real_gest = np.reshape(np.transpose(gest_marks[:,1:]),t.shape)
#%% Plotting
which_elect = 23 #Get activity of electrode #23. Could do any electode here
elec_to_plot = whole_buffer[:,which_elect]
# Spiking data is noisy. Smooth here with a 50 sample window.
elec_to_plot = np.convolve(elec_to_plot,np.ones(50)/50,'same')

# Block start and end for xlimits
block_start = t[block_start_idx]
block_end   = t[block_end_idx]

# Ground-truth/cued gestures, what the patient was instructed and when
plt.subplot(3,1,1)
plt.plot(t,np.float64(real_gest),'r-'),plt.xlim([block_start,block_end])

# Z-scored firing rate from 'which_elect' above
plt.subplot(3,1,2)
plt.plot(t,elec_to_plot),plt.xlim([block_start,block_end])

# Predicted gestures. 
plt.subplot(3,1,3)
plt.plot(t,pred_smooth),plt.xlim([block_start,block_end])

#plt.show
