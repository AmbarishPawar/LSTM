"""
author @ambarish

This function aligns the data by time and electrode

"""
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model

def Format_data_for_LSTM(train_data):
    kernel_size  = 10 #smoothing kernel
    kernel       = np.ones(kernel_size)/kernel_size
    # Choose starting and ending sample for start and end of gesture. These are
    # hyperparameters that were chosen based on classification performance
    start_idx    = 30 
    stop_idx     = 100
    # Initialize arrays
    FR_train     = np.zeros((126,128,70)); 
    labels_train = np.zeros((126,1));
    FR           = np.zeros((126,128,181));
    
    # Get variables from dictionary
    channels     = train_data['chann']
    fr           = train_data['fr_win_nev']
    train_labels = train_data['stim']
    features     = np.unique(channels)
    num_reps     = int(channels.shape[1]/len(features))    
    
    # Go through each trial repetition to align data by rep x electrode x timestamps
    for i in range(0,num_reps):
        fr_this_rep    = fr[i::num_reps,::]
        fr_this_rep_sm = np.zeros(fr_this_rep.shape)
        # Sometimes electrodes are turned off due to presence of 60 Hz
        # noise of jaw clench artifact. So look for NaNs and replace them with
        # mean of firing rates of non-nan electrodes.
        for j in range(0,fr_this_rep.shape[0]):
            check_nan = np.isnan(fr_this_rep[j,:])
            if check_nan.any():
                fr_this_rep[j,:]= np.nanmean(fr_this_rep,0)
            # Smooth data by 'kernel' size. This is another hyperparameter and makes
            # a big difference. 
            fr_this_rep_sm[j,:] = np.convolve(fr_this_rep[j,:],kernel,'same')
            continue 
        # Align the gesture labels             
        labels_this_rep   = train_labels[0,i::num_reps]
        FR_train[i,:,:]   = fr_this_rep_sm[:,start_idx:stop_idx]
        labels_train[i,:] = labels_this_rep[0]
        FR[i,:,:]         = fr_this_rep
        continue
    
    labels_train = np.array(labels_train,dtype=float)    
    FR_train = np.transpose(FR_train,(0,2,1)) #reshape to match what the LSTM needs
    return FR_train, labels_train #return 3-D firing rates and labels