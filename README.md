# LSTM_gesture_decoding

This repo contains python scripts that train a Long Short-Term Memory (LSTM) network to classify hand gestures using neural spiking data obtained from an intracortical Brain Computer Interface (BCI) system.

The repo contains the following scripts:

**Train_LSTM_classify_gestures.py**: This script takes data from a .mat file (Gestures_for_LSTM_train) and trains a two-layer LSTM to classify data into three hand gestures: Rest, Open and Pinch. The performance of the LSTM can be evaluated on 4 other datasets collected on different days. The trained LSTM is saved in the folder 'LSTM_trained_model'

**LSTM_functions_online.py**: As the name suggests, this script contains functions used by other scripts.
- Format_data_for_LSTM: This functions performs much of the processing to arrange the data in the format that can be fed into the LSTM.
- Process_and_classify_per_block: This function classifies neural data into gestures in real-time, as data is live-streamed into the decoder.

**BCI_simulated_expt_realtime.py**: This script does continuous classification of streaming neural spiking data into rest, open and pinch gestures. It simulates an 'online' BCI experiment where data (Online_data.mat) is streamed into the decoder continuously, timestamp-by-timestamp. The trained LSTM model 'LSTM_trained_model', then processes the incoming neural data and classified it into the three gestures.

_Data files_: The repo also contains 5 data files: Gestures_for_LSTM_*.mat. Each file is data from a different experiment performed on different days. While one of the data is explicitly named as 'Gestures_for_LSTM_train', the format of each of the data files is identical, and any one of these files could be used to train and test the LSTM.

The .mat files are preprocessed in Matlab. The raw spike timestamps have been converted to a firing rate, square-rooted and z-scored.
