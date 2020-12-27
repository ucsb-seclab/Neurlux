import keras
from config import *
from Attention import *
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, f1_score, confusion_matrix, recall_score, precision_score
from keras.layers import Conv1D, MaxPooling1D
import sys
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax

#benign_pickle = "ember_lastline_feature_extract/ember_lastline_all_benign.pickle"
#malicious_pickle = "ember_lastline_feature_extract/ember_lastline_all_malicious.pickle"
benign_pickle = "ember_cuckoo_feature_extract/ember_cuckoo_all_benign.pickle"
malicious_pickle = "ember_cuckoo_feature_extract/ember_cuckoo_all_malicious.pickle"

#benign_pickle = "/home/chani/chris/wild_cuckoo_feature_extract/wild_cuckoo_all_benign.pickle"
#malicious_pickle = "/home/chani/chris/wild_cuckoo_feature_extract/wild_cuckoo_all_malicious.pickle"

result_path = "ensemble"
global file_tag
file_tag = "ensemble_train_other"

feature_model_path = "new_att/att_{}.model"
# REMOVED NET
feature_list = ['reg', 'file', 'dll', 'api', 'mutex']

feat_to_maxlen = {
'reg': 5000,
'file': 5000,
'api': 10000,
'dll': 1000,
'mutex': 1000,
'net': 2000,
}

total_maxlen = sum(feat_to_maxlen.values())

epochs = 2
learning_rate=.001

tokenizer_path = "new_att/{}_tokenizer.pickle"

global g_x_test
global g_y_test
g_x_test = None
g_y_test = None


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
        global g_x_test
        global g_y_test
        X_val = g_x_test
        y_val = g_y_test
        #Xa_val, y_val = self.validation_data[0], self.validation_data[1]
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
            'confusion_matrix': confusion_matrix(y_val, y_predict.round())
        })
        return


def load_data(benign_pickle, malicious_pickle, balance=False, limit=None):
    benign = pd.read_pickle(benign_pickle)
    malicious = pd.read_pickle(malicious_pickle)
    df1 = pd.DataFrame(benign)
    df1['label'] = 0
    df2 = pd.DataFrame(malicious)
    df2['label'] = 1

    if balance or limit:
        if balance:
            min_len = min(len(df1), len(df2))
        else:
            min_len = max(len(df1), len(df2))
        if limit is not None:
            min_len = min(min_len, limit)

        # limit it (use sample to sort it first)
        df1 = df1.sample(frac=1)[:min_len]
        df2 = df2.sample(frac=1)[:min_len]

    # concat and rearrange
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1)
    df = df.sample(frac=1)

    # grab feature and label
    x = df
    y = df['label']
    return x, y, benign, malicious


def tokenize_data(x, y):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    vocab_size = len(tokenizer.word_index) + 1
    return x, y, vocab_size, tokenizer


def split_data(x, y):
    num_train = round(0.75*len(x))
    X_train = x[:num_train]
    X_test = x[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    return X_train, X_test, y_train, y_test


def train(model, X_train, X_test, y_train, y_test):
    global g_x_test
    global g_y_test
    g_x_test = X_test
    g_y_test = y_test

    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam',
    #              metrics=['accuracy'])

    loss_history=LossHistory()
    metrics=Metrics()

    ckpt = ModelCheckpoint("{}/att_{}_checkpoint".format(result_path, file_tag), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test), callbacks=[ckpt, early, loss_history, metrics])
    model.save("{}/att_{}.model".format(result_path, file_tag))
    model.save("{}/att_{}_att.model".format(result_path, file_tag))

    model.summary()

    data = {}
    data['metrics'] = metrics._data
    data['acc_history'] = loss_history.accuracy
    data['loss_history'] = loss_history.losses
    with open("{}/{}_att_acc_data.pickle".format(result_path, file_tag), "wb") as f:
        pickle.dump(data, f)

    return history


# with CustomObjectScope({'Attention': Attention()}):

def get_reverse_token_map(tokenizer):
    reverse_token_map = dict(map(reversed, tokenizer.word_index.items()))
    return reverse_token_map

"""
def get_word_importances(text, tokenizer, model, attention_model):
    reverse_token_map = get_reverse_token_map(tokenizer)
    lt = tokenizer.texts_to_sequences([text])
    x = pad_sequences(lt, maxlen=maxlen)
    p = model.predict(x)
    att = attention_model.predict(x)
    return p, [(reverse_token_map.get(word), importance) for word, importance in zip(x[0], att[0]) if word in reverse_token_map]
"""

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('{}/att_{}.png'.format(result_path, file_tag))

def retest(model, pickle_benign, pickle_malicious, save_tag):
    x, y, benign, malicious = load_data(pickle_benign, pickle_malicious, balance=True, limit=1000)
    print("\n\nretest {} {}".format(save_tag, len(x)))

    x = split_and_tokenize(x)
    
    #is_model_present = os.path.isfile('model.hdf5')
    y_predict = model.predict(x)
    data = {}
    data['recall'] = recall_score(y, y_predict.round())
    data['precision'] = precision_score(y, y_predict.round())
    data['accuracy'] = accuracy_score(y, y_predict.round())
    data['f1_score'] = f1_score(y, y_predict.round())
    print("RETEST")
    print(data)
    tn, fp, fn, tp = confusion_matrix(y, y_predict.round()).ravel()
    print("tn: {}    fp: {}\nfn: {}    tp: {}".format(tn, fp, fn, tp))
    print("END RETEST")
    with open("{}/{}_{}_retest".format(result_path,file_tag,save_tag), "w") as f:
        f.write(str(data) + "\n" + "tn fp fn fp\n" + str((tn, fp, fn, tp)) + "\n")
    with open("{}/{}_{}_retest.pickle".format(result_path,file_tag,save_tag), "wb") as f:
        pickle.dump((data,tn, fp, fn, tp), f)

#### NEW ENSEMBLE CODE ########################################################################
 
# load models from file
def load_all_models():
    all_models = list()
    for f in feature_list:
        print("Loading model {}".format(f))
        # define filename for this ensemble
        filename = feature_model_path.format(f)
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models
 
# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    # plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
"""
# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	# X = [inputX[f] for f in feature_list]
	# fit model
	return model.fit(X, inputy, epochs=11, verbose=0)
 
# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
    # X = [inputX[f] for f in feature_list]
	# make prediction
	return model.predict(X, verbose=0)
"""

def split_and_tokenize(x):
    out_x = []
    for f in feature_list:
        with open(tokenizer_path.format(f), "rb") as infile:
            tokenizer = pickle.load(infile)
        tmp = tokenizer.texts_to_sequences(x[f])
        maxlen = feat_to_maxlen[f]
        tmp = pad_sequences(tmp, padding='post', maxlen=maxlen)
        out_x.append(tmp)
    return out_x

if __name__ == '__main__':
    print("loading data")
    x, y, benign, malicious = load_data(benign_pickle, malicious_pickle, balance=True)
    X_train, X_test, y_train, y_test = split_data(x, y)

    print("split and tokenize")
    X_train = split_and_tokenize(X_train)
    X_test = split_and_tokenize(X_test)

    print("HAS {} SUBMODELS".format(len(X_test)))

    models = load_all_models()
    # create stacked model
    model = define_stacked_model(models)

    #history = fit_stacked_model(stacked_model, testX, testy) # don't use
    history = train(model, X_train, X_test, y_train, y_test)

    # evaluate
    retest(model, benign_pickle, malicious_pickle, 'test_verify')
    retest(model, 'ember_cuckoo_feature_extract/ember_cuckoo_all_benign.pickle', 'ember_cuckoo_feature_extract/ember_cuckoo_all_malicious.pickle', 'test_sandbox')
    retest(model, 'wild_lastline_feature_extract/wild_lastline_all_benign.pickle', 'wild_lastline_feature_extract/wild_lastline_all_malicious.pickle', 'test_dataset')
    retest(model, 'wild_cuckoo_feature_extract/wild_cuckoo_all_benign.pickle', 'wild_cuckoo_feature_extract/wild_cuckoo_all_malicious.pickle', 'test_dataset_and_sandbox')

    #malicious_attention = []
    #test_mal = malicious = malicious[:1000]
    #for mal in test_mal:
    #    p, important = get_word_importances(
    #        mal, tokenizer, model, attention_model)
    #    malicious_attention.append(important)
    #with open('{}/malicious_attention_{}.pickle'.format(result_path, file_tag), 'wb') as f:
    #    pickle.dump(malicious_attention, f)

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=True)
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)
    with open("{}/{}_overall".format(result_path, file_tag), "w") as f:
        f.write("Training Accuracy: {:.4f}\n".format(train_accuracy))
        f.write("Testing Accuracy:  {:.4f}\n".format(accuracy))





