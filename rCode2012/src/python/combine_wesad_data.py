'''
Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven. 2018. 
Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. 
In 2018 International Conference on Multimodal Interaction (ICMI â€™18), October 16â€“20, 2018, Boulder, CO, USA. ACM, New York, NY, USA, 9 pages.
'''
# Example script analyzing Wearable Stress and Affect Detection (WESAD) Dataset from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29#

# Import directories
import os
import glob
import pickle
import pandas as pd

'''
According to the README:
The double-tap signal pattern was used to manually synchronise the two devices’ raw data. The result is provided in the files SX.pkl, one file per subject. This file is a dictionary, with the following keys:
- ‘subject’: SX, the subject ID
- ‘signal’: includes all the raw data, in two fields:
o ‘chest’: RespiBAN data (all the modalities: ACC, ECG, EDA, EMG, RESP, TEMP)
o ‘wrist’:EmpaticaE4data(allthemodalities:ACC,BVP,EDA,TEMP)
- ‘label’: ID of the respective study protocol condition, sampled at 700 Hz. The following IDs
are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset
'''

# Study protocal conditions (label) mapping
label_map = {
    0: 'not defined / transient',
    1: 'baseline',
    2: 'stress',
    3: 'amusement',
    4: 'meditation',
}


# Read in WESAD datasets by subject and unpickle
subject_dct = {}
path = '../../data/WESAD'
filenames = glob.glob(os.path.join(path,'*/*.pkl'))
for file in filenames:
    # Had to use 'latin1' as the encoding due to Python 2/3 pickle incompatibility
    # https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    unpickled_file = pickle.load(open(file,'rb'), encoding='latin1')
    # Grab relevant info
    subject_id = unpickled_file['subject']
    print('processing subject',subject_id)
    chest_dct = unpickled_file['signal']['chest']
    wrist_dct = unpickled_file['signal']['wrist']

    # Process the chest dictionary first as it is more straight forward
    # Since the 'ACC' column contains 3 dimensional tuples, it needs to be processed separately due to pandas expecting the same format for all columns
    # Going to create dictionaries without that column to turn into a dataframe, then add the 'ACC' values later
    tmp_chest_dct = dict((k, chest_dct[k].ravel()) for k in list(chest_dct.keys()) if k not in ['ACC'])
    tmp_chest_df = pd.DataFrame(tmp_chest_dct) # Contains everything except ACC
    tmp_acc_df = pd.DataFrame(chest_dct['ACC'],columns=['ACC_X','ACC_Y','ACC_Z']) # Manually declare keys, otherwise shows up as 0,1,2
    final_chest_df = pd.concat([tmp_chest_df,tmp_acc_df],axis=1)

    # Process wrist dictionary, which will take more care because the samplying frequencies were different 
    # Meaning the number of data points collected for each feature is different (higher frequency equals more data points)
    # Basically this one just needs to be processed manually
    wrist_acc_df = pd.DataFrame(wrist_dct['ACC'],columns=['ACC_X','ACC_Y','ACC_Z'])
    wrist_bvp_df = pd.DataFrame(wrist_dct['BVP'],columns=['BVP'])
    wrist_eda_df = pd.DataFrame(wrist_dct['EDA'],columns=['EDA'])
    wrist_temp_df = pd.DataFrame(wrist_dct['TEMP'],columns=['TEMP'])

    # Add labels as a separate object to be returned
    # While the time granularity is the same as the chest data, I'm not sure yet how to use it 
    # So will just keep it separate and add as needed
    labels = pd.Series(unpickled_file['label'],name='label')
    mapped_labels = labels.map(label_map)

    subject_dct[subject_id] = {
        'chest_df': final_chest_df,
        'wrist_dfs': {
            'wrist_acc_df': wrist_acc_df,
            'wrist_bvp_df': wrist_bvp_df,
            'wrist_eda_df': wrist_eda_df,
            'wrist_temp_df': wrist_temp_df,
        },
        'labels': labels,
        'mapped_labels': mapped_labels,
    }

    print(subject_dct)