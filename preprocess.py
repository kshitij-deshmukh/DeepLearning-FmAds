
"""
Created on Sun May 20 12:01:06 2018
@author: kshitij
"""

import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

DATA_PATH = "./fm_dataset/"

# Input: Folder Path
# Output: Tuple(Label, Indices of the labels and one-hot encoded labels)
def get_data_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


#For Padding - put zeros at the end - this wont affect the MFCC
#If you transform back from MFCC to frequency, those zeros wouldn't affect the result. 
# Convert a wave file to mfcc encodings
def to_mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)),
                      mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def save_coefficients(path=DATA_PATH, max_len=11):
    labels, _, _ = get_data_labels(path)

    for label in labels:
        # Init mfcc coefficients
        mfcc_coeffs = []

        wavfiles = [path + label + '/' + wavfile for wavfile in 
                    os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, 
                            "Saving coeffs of label - '{}'".format(label)):
            mfcc = to_mfcc(wavfile, max_len=max_len)
            mfcc_coeffs.append(mfcc)
        np.save(label + '.npy', mfcc_coeffs)

def get_split_data_train_test(split_ratio=0.8, random_state=42):
    # Get labels from the data path
    labels, indices, _ = get_data_labels(DATA_PATH)

    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio),
                            random_state=random_state, shuffle=True)



def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_data_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]