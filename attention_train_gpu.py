import keras
from config import *
from Attention import *
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, f1_score, confusion_matrix, recall_score, precision_score
from keras.layers import Conv1D, MaxPooling1D, CuDNNLSTM, CuDNNGRU
import config

benign_pickle = "reg_benign.pickle"
malicious_pickle = "reg_malicious.pickle"

file_tag = "reg_retry"
batch_size = 128
maxlen = reg_maxlen
maxlen = 10000
epochs = 2
FEATURE="reg"

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
            'confusion_matrix': confusion_matrix(y_val, y_predict.round())
        })
        #IPython.embed()
        return


def load_data(benign_pickle, malicious_pickle):
    benign = pd.read_pickle(benign_pickle)
    malicious = pd.read_pickle(malicious_pickle)

    if isinstance(benign[0], (str, bytes)):
        benign_text = benign
    else:
        benign_text = []
        for i in benign:
            sen = ", ".join(x for x in i)
            benign_text.append(sen)

    if isinstance(malicious[0], (str, bytes)):
        mal_text = malicious
    else:
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
    return x, y, benign, malicious

def load_data2(benign_pickle, malicious_pickle, balance=False, limit=None, no_null=False):
    benign = pd.read_pickle(benign_pickle)
    malicious = pd.read_pickle(malicious_pickle)
    df1 = pd.DataFrame(benign)
    df1['label'] = 0
    df2 = pd.DataFrame(malicious)
    df2['label'] = 1

    if no_null:
        df1 = df1[df1[FEATURE] != ""]
        df2 = df2[df2[FEATURE] != ""]

    if balance or limit:
        if balance:
            min_len = min(len(df1), len(df2))
        else:
            min_len = max(len(df1), len(df2))
        if limit is not None:
            min_len = min(min_len, limit)

        # limit it (use sample to randomly sort it first)
        df1 = df1.sample(frac=1)[:min_len]
        df2 = df2.sample(frac=1)[:min_len]
        #df1 = df1[:min_len]
        #df2 = df2[:min_len]

    # count nulls of each
    null_count_b = len(df1[df1[FEATURE] == ""])
    null_count_m = len(df2[df2[FEATURE] == ""])

    # concat and rearrange
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1)

    # grab feature and label
    x = df[FEATURE]
    y = df['label']
    return x, y, benign, malicious, null_count_b, null_count_m

def retest2(tokenizer, model, pickle_benign, pickle_malicious, save_tag, no_null=False):
    x, y, benign, malicious, null_count_b, null_count_m = load_data2(pickle_benign, pickle_malicious, balance=True, limit=1000, no_null=no_null)
    print("\n\nretest2 {} {}".format(save_tag, len(x)))
    print("nulls in benign {}    nulls in malicious {}".format(null_count_b, null_count_m))
    x = tokenizer.texts_to_sequences(x)

    x = pad_sequences(x, padding='post', maxlen=maxlen)
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

def tokenize_data(x, y):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    vocab_size = len(tokenizer.word_index) + 1
    return x, y, vocab_size, tokenizer


def split_data(x, y):
    x = pad_sequences(x, padding='post', maxlen=maxlen)
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(x), np.array(y), test_size=0.25, random_state=1000)

    return X_train, X_test, y_train, y_test


def get_model(vocab_size):
    inp = Input(shape=(maxlen, ))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=maxlen)(inp)
    print("Input to LSTM dim", x.shape)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4)(x)
    #x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)
    # x shape will be (?, ?, 2x32) (bidirectional doubles the first arg of LSTM)
    print("Input to Attention shape:", x.shape)
    x, attention_out = Attention(name='attention_vec')(x)
    print("Output of Attention shape:", attention_out.shape)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    attention_model = Model(inputs=inp, outputs=attention_out)

    return model, attention_model


def train(model, attention_model, X_train, X_test, y_train, y_test):

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    loss_history=LossHistory()
    metrics=Metrics()

    ckpt = ModelCheckpoint("att_{}_checkpoint".format(file_tag), monitor='val_loss', verbose=1,
                           save_best_only=True, mode='min')
    #FIXME early
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test), callbacks=[ckpt, early, loss_history, metrics])
    model.save("att_{}.model".format(file_tag))
    attention_model.save("att_{}_att.model".format(file_tag))

    model.summary()

    data = {}
    data['metrics'] = metrics._data
    data['acc_history'] = loss_history.accuracy
    data['loss_history'] = loss_history.losses
    with open("{}_att_acc_data.pickle".format(file_tag), "wb") as f:
        pickle.dump(data, f)

    return history


# with CustomObjectScope({'Attention': Attention()}):

def get_reverse_token_map(tokenizer):
    reverse_token_map = dict(map(reversed, tokenizer.word_index.items()))
    return reverse_token_map


def get_word_importances(text, tokenizer, model, attention_model):
    reverse_token_map = get_reverse_token_map(tokenizer)
    lt = tokenizer.texts_to_sequences([text])
    x = pad_sequences(lt, maxlen=maxlen)
    p = model.predict(x)
    att = attention_model.predict(x)
    return p, [(reverse_token_map.get(word), importance) for word, importance in zip(x[0], att[0]) if word in reverse_token_map]


def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('att_{}.png'.format(file_tag))

def retest(tokenizer, model, pickle_benign, pickle_malicious):
    x, y, benign, maliciou = load_data(pickle_benign, pickle_malicious)
    print("len retest {}".format(len(x)))

    x = tokenizer.texts_to_sequences(x)
    
    x = pad_sequences(x, padding='post', maxlen=maxlen)
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


if __name__ == '__main__':
    x, y, benign, malicious = load_data(benign_pickle, malicious_pickle)
    x, y, vocab_size, tokenizer = tokenize_data(x, y)

    with open('{}_tokenizer.pickle'.format(file_tag), 'wb') as f:
             pickle.dump(tokenizer, f)

    X_train, X_test, y_train, y_test = split_data(x, y)

    # model = load_model('model.hdf5')
    model, attention_model = get_model(vocab_size)
    history = train(model, attention_model,
                    X_train, X_test, y_train, y_test)

    # evaluate
    print("different dataset (original comparison)")
    retest(tokenizer, model, 'ember_cuckoo_reg_benign.pickle', 'ember_cuckoo_reg_malicious.pickle')

    # evaluate
    print("verify my extract works")
    retest2(tokenizer, model, 'wild_cuckoo_feature_extract/wild_cuckoo_all_benign.pickle', 'wild_cuckoo_feature_extract/wild_cuckoo_all_malicious.pickle', 'test_verify_no_null', no_null=True)
    retest2(tokenizer, model, 'wild_cuckoo_feature_extract/wild_cuckoo_all_benign.pickle', 'wild_cuckoo_feature_extract/wild_cuckoo_all_malicious.pickle', 'test_verify')
    print("check different dataset")
    retest2(tokenizer, model, 'ember_cuckoo_feature_extract/ember_cuckoo_all_benign.pickle', 'ember_cuckoo_feature_extract/ember_cuckoo_all_malicious.pickle', 'test_dataset_no_null', no_null=True)
    retest2(tokenizer, model, 'ember_cuckoo_feature_extract/ember_cuckoo_all_benign.pickle', 'ember_cuckoo_feature_extract/ember_cuckoo_all_malicious.pickle', 'test_dataset')


    #malicious_attention = []
    #test_mal = malicious = malicious[:1000]
    #for mal in test_mal:
    #    p, important = get_word_importances(
    #        mal, tokenizer, model, attention_model)
    #    malicious_attention.append(important)
    #with open('malicious_attention_{}.pickle'.format(file_tag), 'wb') as f:
    #    pickle.dump(malicious_attention, f)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)
