import pandas as pd
import numpy as np
import json
import random
from neuralnet import NeuralNet
from keras.preprocessing.sequence import pad_sequences

csv_path = None
results_path = None

RANDOM_SEED = 17
np.random.seed(seed=RANDOM_SEED)
EPOCH = 10
BATCH_SIZE = 256
KFOLD = 5


def main(input_path):

    def right_pad(x, MAX_LENGTH):
        r = np.zeros(MAX_LENGTH)
        r[:x.shape[0]] = x
        return r
    
    def invert(x):
        return int(not x)

    # read and divide dataframe
    df = pd.read_pickle(input_path) # dataframe from ;prepare.py; extract_data.py;
    print('loaded pickle file, {}'.format(len(df)))
    df = df.dropna()
    MAX_LENGTH = 1 * 1024 * 128
    #MAX_LENGTH = df['content'].map(len).max()
    df['size'] = df['content'].map(len)
    df = df[df['size'] <= MAX_LENGTH]
    #df = df[len(df['content']) <= MAX_LENGTH]
    print('Length of dataframe after dropping none columns {}'.format(len(df)))
    df['content'] = df['content'].apply(right_pad, MAX_LENGTH = MAX_LENGTH)

    print(len(df))
    benign      = df[df.benign == 1]
    malicious   = df[df.benign == 0]
    print('benign:', benign.shape[0])
    print('malicious:', malicious.shape[0])

    # just shuffle training dataset.
    df = df.sample(frac=1, random_state=RANDOM_SEED)
    # labels are being malicious or benign
    df['benign'] = df['benign'].map(int)
    df['malicious'] = df['benign'].apply(invert)
    y = df[['benign']].values
    
    # remove labels related to packing and type of binary
    x = df['content'].values
    # unpack np ndarray obj into boxed arrays for keras
    print('started rearrangin')

    #x = x.reshape((x.shape[0], MAX_LENGTH))
    #x = np.concatenate(x, axis=0).reshape((x.shape[0], MAX_LENGTH))
    
    x =  pad_sequences(x, padding='post', maxlen=MAX_LENGTH)
    import IPython
    IPython.embed()

    print('finished rearrangin')

    # now train on the whole dataset which has been used in CV
    NN = NeuralNet(RANDOM_SEED)
    NN.cross_validation(x, y, KFOLD, EPOCH, BATCH_SIZE)
    NN.save_model("raw_model")
    NN.model.save("raw_model_chris")
    import IPython; IPython.embed()

if __name__ == '__main__':
    main('withcontents.pickle')

