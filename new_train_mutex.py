import keras
import pickle
import IPython
import numpy as np
import pandas as pd
import keras_metrics
import plotly.offline as py
import keras.layers as layers
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from keras.models import Model
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Conv1D, Embedding, Dropout, Flatten
from keras.layers import Dense, Activation, Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score
from keras.layers import Dense, Input, GlobalMaxPooling1D, concatenate, MaxPooling1D
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, f1_score

py.init_notebook_mode(connected=True)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_acc'))


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))

        # precision recall is [precision], [recall], [thresh]
        # roc is [tpr], [fpr], [thresh]
        self._data.append({
            'val_recall': recall_score(y_val, y_predict.round()),
            'val_precision': precision_score(y_val, y_predict.round()),
            'val_accuracy': accuracy_score(y_val, y_predict.round()),
            'val_precision_recall_curve': precision_recall_curve(y_val, y_predict),
            'val_roc_curve': roc_curve(y_val, y_predict),
            'val_f1_score': f1_score(y_val, y_predict.round()),
        })
        #IPython.embed()
        return

epochs = 10
embedding_dim = 200
maxlen = 5000
output_file = 'data/output.txt'

file_tag = "mutex"
benign = pd.read_pickle('mutex_benign.pickle')
malicious = pd.read_pickle('mutex_malicious.pickle')

benign_text = []
for i in benign:
    sen = ", ".join(x for x in i)
    benign_text.append(sen)

mal_text = []
for i in malicious:
    sen = ", ".join(x for x in i)
    mal_text.append(sen)

df1 = pd.DataFrame(benign_text, columns=['data'])
df1['label'] = 0

df2 = pd.DataFrame(mal_text, columns=['data'])
df2['label'] = 1

df = pd.concat([df1, df2], ignore_index=True)

df = df.sample(frac=1)

x = df['data']
y = df['label']


#for k, v in sorted(tokenizer.word_counts.items(), key = lambda x:-x[1]):
tokenizer_benign = Tokenizer(num_words=3000)
tokenizer_mal = Tokenizer(num_words=3000)
tokenizer_benign.fit_on_texts(df1['data'])
tokenizer_mal.fit_on_texts(df2['data'])

most_common_benign = []
most_common_mal = []

count = 0
for k, v in sorted(tokenizer_benign.word_counts.items(), key = lambda x:-x[1]):
    most_common_benign.append(k)
    count += 1
    if count >= 200:
       break

count = 0
for k, v in sorted(tokenizer_mal.word_counts.items(), key = lambda x:-x[1]):
    most_common_mal.append(k)
    count += 1
    if count >= 200:
       break

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

x = pad_sequences(x, padding='post', maxlen=maxlen)
print(len(x))

X_train, X_test, y_train, y_test = train_test_split(
    np.array(x), np.array(y), test_size=0.25, random_state=1000)


model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(512, 5, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Conv1D(256, 5, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.GlobalMaxPooling1D())
#model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

loss_history=LossHistory()
metrics=Metrics()
history = model.fit(np.array(X_train), np.array(y_train),
                    epochs=10,
                    verbose=True,
                    validation_data=(np.array(X_test), np.array(y_test)),
                    batch_size=10,
                    callbacks=[loss_history, metrics])

data = {}
data['metrics'] = metrics._data
data['acc_history'] = loss_history.accuracy
data['loss_history'] = loss_history.losses
with open("{}_acc_data.pickle".format(file_tag), "wb") as f:
    pickle.dump(data, f)

#IPython.embed()

conv_embds = model.layers[0].get_weights()[0]

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(conv_embds):
    print(len(conv_embds))
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in most_common_benign:
         #print("index:", index)
         print("conv_embed size:", len(conv_embds))
         tokens.append(conv_embds[tokenizer.word_index.get(word)])
         labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i], color='red')
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
    plt.show()
    return plt


def tsne_plot2(conv_embds, plt):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in most_common_mal:
#         print("index:", index)
#         print("conv_embed size:", len(conv_embds))
        tokens.append(conv_embds[tokenizer.word_index.get(word)])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    for i in range(len(x)):
        plt.scatter(x[i],y[i], color='blue')
#         plt.annotate(labels[i],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
    plt.savefig('{}_tsne_most_common.png'.format(file_tag))
    plt.clf()
    

conv_embds = model.layers[0].get_weights()
plt = tsne_plot(conv_embds[0])
tsne_plot2(conv_embds[0], plt)


loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#print('hist')
#print(history.history['acc'])
#print(history.history['val_acc'])

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_cnn_history.png'.format(file_tag))
    plt.clf()

plot_history(history)

model.save("{}_trained.model".format(file_tag))

