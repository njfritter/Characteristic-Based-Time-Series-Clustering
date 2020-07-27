# Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven. 2018. Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. In 2018 International Conference on Multimodal Interaction (ICMI â€™18), October 16â€“20, 2018, Boulder, CO, USA. ACM, New York, NY, USA, 9 pages.

# Example script analyzing Wearable Stress and Affect Detection (WESAD) Dataset from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer

# Import directories
import glob
import pandas as pd

# Read in WESAD datasets by subject and unpickle
subject_dct = {}
path = '../../data/WESAD'
filenames = glob.glob(path + '/*/*.pkl')
for file in filenames:
    # Had to use 'latin1' as the encoding due to Python 2/3 pickle incompatibility
    # https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    unpickled_file = pickle.load(open(file,'rb'), encoding='latin1')
    subject_dct[unpickled_file['subject']] = unpickled_file

'''
Accoridng to the README:
The double-tap signal pattern was used to manually synchronise the two devices’ raw data. The result is provided in the files SX.pkl, one file per subject. This file is a dictionary, with the following keys:
- ‘subject’: SX, the subject ID
- ‘signal’: includes all the raw data, in two fields:
o ‘chest’: RespiBAN data (all the modalities: ACC, ECG, EDA, EMG, RESP, TEMP)
o ‘wrist’:EmpaticaE4data(allthemodalities:ACC,BVP,EDA,TEMP)
- ‘label’: ID of the respective study protocol condition, sampled at 700 Hz. The following IDs
are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset
'''

