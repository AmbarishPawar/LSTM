"""
@author: ambarish
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tensorflow import keras

model = keras.models.load_model('LSTM_model_sm')
#%% Load data from block
data         = loadmat('online_data.mat')

#%% Process data as if it is streaming from the brain
fr           = data['fr']
gest_marks   = data['gest_marks']
real_timebin = data['real_timebin']
cue_on_nev   = data['cue_on_nev']
cue_off_nev  = data['cue_off_nev']

decode_win  = 30
z_buffer    = np.zeros((2000,128))
time_buffer = np.zeros((1,2000))
fr_buffer   = np.full([2000,128],np.nan) #Create array of nans
zscore_part = np.zeros((1,decode_win,128))

used_nsp     = 2
fr_nsp       = fr[used_nsp,:,:]
whole_buffer = np.zeros((fr_nsp.shape[1],128))
fr_sqrt      = np.zeros((fr_nsp.shape[0],)) #one dimensional
YPred        = np.zeros((fr_nsp.shape[1],3))
YClass       = np.zeros((fr_nsp.shape[1],))
pred_smooth  = np.zeros((fr_nsp.shape[1],))
bins         = [0,1,2,3]
thresh       = 0.8

for counter_packets in range(0,fr_nsp.shape[1]):
    print(counter_packets)
    fr_sqrt = np.sqrt(fr_nsp[:,counter_packets])
    fr_buffer = np.roll(fr_buffer,-1,0)
    time_buffer = np.roll(time_buffer,-1,1)
    
    fr_buffer[-1,:] = fr_sqrt
    time_buffer[-1] = real_timebin[0,counter_packets]
    fr_mn = np.nanmean(fr_buffer,0)
    fr_sd = np.nanstd(fr_buffer,0,ddof=1)
    
    if counter_packets>0:
        z_score = ((fr_sqrt-fr_mn)/fr_sd)
        z_score[np.isnan(z_score)] = 0
        z_score[np.isinf(z_score)] = 0
        
        z_buffer = np.roll(z_buffer,-1,0)
        z_buffer[-1,:] = z_score
        
    else:
         z_buffer[-1,:] = (fr_sqrt-fr_mn)/2
    
    whole_buffer = np.roll(whole_buffer,-1,0)
    whole_buffer[-1,:] = z_buffer[-1,:]
    
    zscore_part[0,:,:] = z_buffer[-decode_win:,:]
    t_part = time_buffer[-decode_win:,:]
    Y = model.predict(zscore_part)
    YPred[counter_packets,:] = Y
    YClass[counter_packets] = Y.argmax(axis=-1)
    
    if counter_packets > (decode_win-1):
        classify_in_window,b = np.histogram(YClass[counter_packets-decode_win:counter_packets],bins)    
        max_class = np.max(classify_in_window)
        max_class_idx = np.argmax(classify_in_window)
        if max_class>=(thresh*decode_win):
            pred_smooth[counter_packets] = max_class_idx
        else:
            pred_smooth[counter_packets] = 0
    continue

time_bins_block = [cue_on_nev[1]-0.5,cue_off_nev[-1]+3]
block_start_idx = np.abs(real_timebin-time_bins_block[0]).argmin();
block_end_idx   = np.abs(real_timebin-time_bins_block[-1]).argmin();

#%% Getting classification accuracy
reaction_delay = 0
decode_delay   = 0.35
total_delay    = reaction_delay+decode_delay
decode_end     = total_delay+2.5

t = real_timebin
t = np.reshape(t,t.shape[1],)

YMov = np.zeros(len(cue_on_nev))
for c in range(len(cue_on_nev)):
    this_cue_on = cue_on_nev[c]
    this_cue_off = this_cue_on+decode_end
    cue_idx = np.where(np.logical_and(t>=(this_cue_on+total_delay),t<=this_cue_off))
    this_pred = pred_smooth[cue_idx]
    classified_as_what,temp1 = np.histogram(this_pred,[0,1,2,3])
    max_count_idx = np.argmax(classified_as_what)
    YMov[c] = max_count_idx

t = t[1:]
real_gest = np.reshape(np.transpose(gest_marks[:,1:]),t.shape)
#%% Plotting
which_elect = 23
elec_to_plot = whole_buffer[:,which_elect]
elec_to_plot = np.convolve(elec_to_plot,np.ones(50)/50,'same')

block_start = t[block_start_idx]
block_end   = t[block_end_idx]

plt.subplot(3,1,1)
plt.plot(t,np.float64(real_gest),'r-'),plt.xlim([block_start,block_end])

plt.subplot(3,1,2)
plt.plot(t,elec_to_plot),plt.xlim([block_start,block_end])

plt.subplot(3,1,3)
plt.plot(t,pred_smooth),plt.xlim([block_start,block_end])

#plt.show
