"""
author @ambarish

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tensorflow import keras

""" This function aligns data by time and electrode"""
def Format_data_for_LSTM(train_data):
    kernel_size  = 10 #smoothing kernel, to smooth firing rate data
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
                
            # Smooth data by 'kernel' size. This is another hyperparameter and makes a big difference. 
            fr_this_rep_sm[j,:] = np.convolve(fr_this_rep[j,:],kernel,'same')
             
        # Align the gesture labels             
        labels_this_rep   = train_labels[0,i::num_reps]
        FR_train[i,:,:]   = fr_this_rep_sm[:,start_idx:stop_idx]
        labels_train[i,:] = labels_this_rep[0]
        FR[i,:,:]         = fr_this_rep        
    
    labels_train = np.array(labels_train,dtype=float)    
    FR_train = np.transpose(FR_train,(0,2,1)) #reshape to match what the LSTM needs
    
    return FR_train, labels_train #return 3-D firing rates and labels

"""
This function classifies neural data into gestures in real-time, as data is
live-streamed into the decoder
"""
def Process_and_classify_per_block(model,data,decode_win,threshold):
    
    #%% Process data as if it is streaming from the brain
    fr           = data['fr']           #Firing rate
    gest_marks   = data['gest_marks']   #Which gesture in real-time
    real_timebin = data['real_timebin'] #Actual timeline in seconds
    cue_on_nev   = data['cue_on_nev']   #Timestamp of when gesture started
    cue_off_nev  = data['cue_off_nev']  #Timestamp of when gesture ended
    gestureLabels= data['gestureLabels']#Ground-truth gesture per-trial
    
    # Initialize buffers and other variables
    z_buffer    = np.zeros((2000,128)) #z-score buffer. Larger buffer
    time_buffer = np.zeros((1,2000))
    fr_buffer   = np.full([2000,128],np.nan) #Create array of nans
    zscore_part = np.zeros((1,decode_win,128))
    
    used_nsp     = 2 #nsp: neural signnal processor. Neural data from right-hemisphere, left hand gestures
    fr_nsp       = fr[used_nsp,:,:]
    whole_buffer = np.zeros((fr_nsp.shape[1],128))
    fr_sqrt      = np.zeros((fr_nsp.shape[0],))
    YPred        = np.zeros((fr_nsp.shape[1],3))
    YClass       = np.zeros((fr_nsp.shape[1],))
    pred_gesture = np.zeros((fr_nsp.shape[1],)) 
    bins         = [0,1,2,3]
    
    for counter_packets in range(0,fr_nsp.shape[1]):
        #print(counter_packets) #Only to check progress of code. 
        fr_sqrt     = np.sqrt(fr_nsp[:,counter_packets]) #square-root of firing rate
        fr_buffer   = np.roll(fr_buffer,-1,0) #shift rows, to accomodate new data
        time_buffer = np.roll(time_buffer,-1,1)
        
        fr_buffer[-1,:] = fr_sqrt
        time_buffer[-1] = real_timebin[0,counter_packets]
        
        # Mean and std taken to calculate z-score
        fr_mn = np.nanmean(fr_buffer,0) 
        fr_sd = np.nanstd(fr_buffer,0,ddof=1)
        
        # If buffer has only 1 column, std of one column will be zero, and z-score
        # calculation will yield NaNs. This loop gets over that condition.
        if counter_packets>0:
            z_score = ((fr_sqrt-fr_mn)/fr_sd)
            z_score[np.isnan(z_score)] = 0 #remove NaNs
            z_score[np.isinf(z_score)] = 0 #remove inf
            
            z_buffer = np.roll(z_buffer,-1,0) #shift rows up to accomodate new incoming data
            z_buffer[-1,:] = z_score #add new data
            
        else:
            # Do not divide by std, otherwise we'll get only NaNs
             z_buffer[-1,:] = (fr_sqrt-fr_mn)/2
        
        # whole_buffer stored all z-scored data, so that we can examine it later if we wanted to. 
        whole_buffer = np.roll(whole_buffer,-1,0) #shift rows for new data
        whole_buffer[-1,:] = z_buffer[-1,:] #add new data
        
        # zscore_part only takes 'decode_win' number of samples that will be fed to the decoder. 
        zscore_part[0,:,:] = z_buffer[-decode_win:,:]
        t_part = time_buffer[-decode_win:,:]
        # Predict the gesture based on 'decode_win' amount of z-scored data
        Y = model.predict(zscore_part)
        YPred[counter_packets,:] = Y #store probabilities
        YClass[counter_packets] = Y.argmax(axis=-1)#convert probabilities to integer states
        
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
                #if more than threshold, this is predicted gesture
                pred_gesture[counter_packets] = max_class_idx
            else:
                pred_gesture[counter_packets] = 0 #default to rest gesture
        continue
    
    # We want to find when the block actually starts. Get the times here. 
    time_bins_block = [cue_on_nev[0]-0.5,cue_off_nev[-1]+3]
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
    # the LSTM decoder. These need to be taken into account before calculating accuracy.
    YMov = np.zeros(len(cue_on_nev))
    for c in range(len(cue_on_nev)):
        this_cue_on = cue_on_nev[c]
        this_cue_off = this_cue_on+decode_end
        #(For later) needs to be converted to np array and reshaped to index into it
        cue_idx = np.where(np.logical_and(t>=(this_cue_on+total_delay),t<=this_cue_off))
        this_pred = pred_gesture[cue_idx]
        classified_as_what,temp1 = np.histogram(this_pred,[0,1,2,3])
        max_count_idx = np.argmax(classified_as_what)
        YMov[c] = max_count_idx
    
    # Simple reshaping to match dimensions for plotting
    t = t[1:]    
    
    # Block start and end for xlimits
    block_start = t[block_start_idx]
    block_end   = t[block_end_idx]
    
    return YMov,gestureLabels,real_timebin,gest_marks,whole_buffer,pred_gesture,block_start,block_end,t
