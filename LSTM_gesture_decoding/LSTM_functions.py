import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model

def Format_data_for_LSTM(train_data):
    #!cls
    kernel_size  = 10
    kernel       = np.ones(kernel_size)/kernel_size
    start_idx    = 30 
    stop_idx     = 100
    FR_train     = np.zeros((126,128,70)); 
    labels_train = np.zeros((126,1));
    FR           = np.zeros((126,128,181));
    
    # get variables fromm dictionary
    channels     = train_data['chann']
    fr           = train_data['fr_win_nev']
    train_labels = train_data['stim']
    features     = np.unique(channels)
    num_reps     = int(channels.shape[1]/len(features))    
    
    for i in range(0,num_reps):
        fr_this_rep    = fr[i::num_reps,::]
        fr_this_rep_sm = np.zeros(fr_this_rep.shape)
        for j in range(0,fr_this_rep.shape[0]):
            check_nan = np.isnan(fr_this_rep[j,:])
            if check_nan.any():
                fr_this_rep[j,:]= np.nanmean(fr_this_rep,0)
            fr_this_rep_sm[j,:] = np.convolve(fr_this_rep[j,:],kernel,'same') #smoothing goes here   
            
            continue              
        labels_this_rep = train_labels[0,i::num_reps]
        FR_train[i,:,:] = fr_this_rep_sm[:,start_idx:stop_idx]
        #FR_train.append(fr_this_rep[:,start_idx:stop_idx])
        #labels_train.append(labels_this_rep[0])
        labels_train[i,:] = labels_this_rep[0]
        FR[i,:,:] = fr_this_rep
        continue
    labels_train = np.array(labels_train,dtype=float)    
    FR_train = np.transpose(FR_train,(0,2,1))
    return FR_train, labels_train


def show_features_1D(model=None, layer_name=None, input_data=None,
                     prefetched_outputs=None, max_timesteps=100,
                     max_col_subplots=10, equate_axes=False,
                     show_y_zero=True, channel_axis=-1,
                     scale_width=1, scale_height=1, dpi=76):
    if prefetched_outputs is None:
        layer_outputs = get_layer_outputs(model, layer_name, input_data, 1)[0]
    else:
        layer_outputs = prefetched_outputs
    n_features    = layer_outputs.shape[channel_axis]

    for _int in range(1, max_col_subplots+1):
      if (n_features/_int).is_integer():
        n_cols = int(n_features/_int)
    n_rows = int(n_features/n_cols)

    fig, axes = plt.subplots(n_rows,n_cols,sharey=equate_axes,dpi=dpi)
    fig.set_size_inches(24*scale_width,16*scale_height)

    subplot_idx = 0
    for row_idx in range(axes.shape[0]):
      for col_idx in range(axes.shape[1]): 
        subplot_idx += 1
        feature_output = layer_outputs[:,subplot_idx-1]
        feature_output = feature_output[:max_timesteps]
        ax = axes[row_idx,col_idx]

        if show_y_zero:
            ax.axhline(0,color='red')
        ax.plot(feature_output)

        ax.axis(xmin=0,xmax=len(feature_output))
        ax.axis('off')

        ax.annotate(str(subplot_idx),xy=(0,.99),xycoords='axes fraction',
                    weight='bold',fontsize=14,color='g')
    if equate_axes:
        y_new = []
        for row_axis in axes:
            y_new += [np.max(np.abs([col_axis.get_ylim() for 
                                     col_axis in row_axis]))]
        y_new = np.max(y_new)
        for row_axis in axes:
            [col_axis.set_ylim(-y_new,y_new) for col_axis in row_axis]
    plt.show()
    
    
    
def show_features_2D(data, cmap='bwr', norm=None,
                     scale_width=1, scale_height=1):
    if norm is not None:
        vmin, vmax = norm
    else:
        vmin, vmax = None, None  # scale automatically per min-max of 'data'

    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel('Timesteps', weight='bold', fontsize=14)
    plt.ylabel('Attention features', weight='bold', fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)  # works for any size plot

    plt.gcf().set_size_inches(8*scale_width, 8*scale_height)
    plt.show()
    
    
def get_layer_outputs(model, layer_name, input_data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    layer_outputs = intermediate_layer_model.predict(input_data)
    return layer_outputs