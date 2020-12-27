import IPython
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers import Conv1D, Embedding, Dropout, Flatten
from keras.layers import Dense, Input, GlobalMaxPooling1D, concatenate, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Concatenate
import matplotlib.pyplot as plt
import keras.layers as layers
import numpy as np
import pandas as pd
import keras_metrics
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.manifold import TSNE
import plotly.offline as py
import plotly.graph_objs as go
import keras
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

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_recall': recall_score(y_val, y_predict),
            'val_precision': precision_score(y_val, y_predict),
        })
        IPython.embed()
        return

epochs = 20
embedding_dim = 300
maxlen = 5000
output_file = 'data/output.txt'

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

#IPython.embed()
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

#IPython.embed()

x = pad_sequences(x, padding='post', maxlen=maxlen)
print(len(x))

X_train, X_test, y_train, y_test = train_test_split(
    np.array(x), np.array(y), test_size=0.25, random_state=1000)


model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(layers.Conv1D(128, 7, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.5))
model.add(layers.Conv1D(256, 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.5))
model.add(layers.Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.5))
model.add(Flatten())
#model.add(layers.GlobalMaxPooling1D())
#model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history=LossHistory()
metrics=Metrics()
history = model.fit(np.array(X_train), np.array(y_train),
                    epochs=2,
                    verbose=True,
                    validation_data=(np.array(X_test), np.array(y_test)),
                    batch_size=10)
                    #callbacks=[history, metrics])

IPython.embed()

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
    plt.show()
    plt.savefig('tsne_most_common_chris.png')
    

conv_embds = model.layers[0].get_weights()
plt = tsne_plot(conv_embds[0])
tsne_plot2(conv_embds[0], plt)


loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
