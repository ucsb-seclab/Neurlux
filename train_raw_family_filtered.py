import pandas as pd
import numpy as np
import json
import random
import neuralnet2
from neuralnet2 import NeuralNet
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, f1_score, confusion_matrix, recall_score, precision_score

csv_path = None
results_path = None

RANDOM_SEED = 17
np.random.seed(seed=RANDOM_SEED)
EPOCH = 3
BATCH_SIZE = 256
KFOLD = 5

MAX_LENGTH = 1 * 1024 * 300
neuralnet2.MAX_LENGTH = MAX_LENGTH

def main(input_path):

    def right_pad(x):
        r = np.zeros(MAX_LENGTH)
        r[:x.shape[0]] = x
        return r
    
    def invert(x):
        return int(not x)

    # read and divide dataframe
    df = pd.read_pickle(input_path) # dataframe from ;prepare.py; extract_data.py;
    print('loaded pickle file, {}'.format(len(df)))
    df = df.dropna()
    print('Length of dataframe after dropping none columns {}'.format(len(df)))
    #MAX_LENGTH = df['content'].map(len).max()
    df['size'] = df['content'].map(len)
    df = df[df['size'] <= MAX_LENGTH]
    #df = df[len(df['content']) <= MAX_LENGTH]
    df['content'] = df['content'].apply(right_pad)
    print('Length of dataframe after dropping large columns {}'.format(len(df)))
 

    import IPython; IPython.embed()
    # just shuffle training dataset.
    df = df.sample(frac=1, random_state=RANDOM_SEED)

    benign      = df[df.benign == 1]
    malicious   = df[df.benign == 0]
    print("before balancing")
    print('benign:', benign.shape[0])
    print('malicious:', malicious.shape[0])

    to_use = min(len(benign), len(malicious))
    benign = benign[:to_use]
    malicious = malicious[:to_use]

    df = benign.append(malicious)
    print("After balancing we have {} samples".format(len(df)))

    # shuffle training dataset.
    df = df.sample(frac=1, random_state=RANDOM_SEED)
 
    # labels are being malicious or benign
    df['benign'] = df['benign'].map(int)
    df['malicious'] = df['benign'].apply(invert)
    y = df[['malicious']].values
    
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
    NN.save_model("raw_model_family_filtered")
    NN.model.save("raw_model_family_filtered_chris")
    import IPython; IPython.embed()

if __name__ == '__main__':
    main('cuckoo_wild_family_filtered_withcontents.pickle')

