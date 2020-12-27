import IPython
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Permute, RepeatVector, LSTM, Bidirectional, Multiply, Lambda, Dense, Dropout, \
    Input,Flatten,Embedding
from keras.callbacks import History, CSVLogger, ModelCheckpoint, EarlyStopping
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, load_model
from keras.utils import CustomObjectScope
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

epochs = 5
embedding_dim = 256

#Depends on the feature
maxlen = 87000

reg_maxlen = 35000
file_maxlen = 10000
api_maxlen = 35000
dll_maxlen = 2000
mutex_maxlen = 2000
net_maxlen = 3000

max_features = 100000
vocab_size = 10000

batch_size = 30

