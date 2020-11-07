'''
# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import pickle

# # df = pd.read_csv('sentiment.csv', header=None, sep=',', encoding="ISO-8859-1", engine='python', error_bad_lines=False)
# df.tail()

# sent_n = pd.DataFrame(df[df[0] == 0][5])
# sent_p = pd.DataFrame(df[df[0] == 4][5])

# sent_n['sentiment'] = 0
# sent_p['sentiment'] = 1

# sent_n.columns = ['content', 'sentiment']
# sent_p.columns = ['content', 'sentiment']

# # sent_p = sent_p[:266000]
# # sent_n = sent_n[:266000]

# # Append all datasets in one big dataset
# dataset = sent_p.append(sent_n, ignore_index=True, sort=True)

# dataset['content'] = dataset['content'].astype('str')
dataset_1 = pd.read_csv('data/final.csv')
dataset_1['sent'].astype(np.int64)
dataset_1 = dataset_1.drop(columns=['Unnamed: 0', 'id'])
dataset_1.columns = ['content', 'sentiment']

neg = pd.read_csv('data/processedNegative.csv', header=None)
neu = pd.read_csv('data/processedNeutral.csv', header=None)
pos = pd.read_csv('data/processedPositive.csv', header=None)

pos = pos.T
neu = neu.T
neg = neg.T

pos['sentiment'] = 2
neu['sentiment'] = 1
neg['sentiment'] = 0

pos.columns = ['content', 'sentiment']
neu.columns = ['content', 'sentiment']
neg.columns = ['content', 'sentiment']

dataset = pos.append([neu, neg, dataset_1], ignore_index=True, sort=True)

dataset['content'] = dataset['content'].astype('str')
dataset['sentiment'] = dataset['sentiment'].astype('int')

c, u  = np.unique(dataset['sentiment'], return_counts=True)
print(dict(zip(c, u)))

samples = int(min(u)//2.5)
print(samples)
print(dataset.head())
dataset = dataset.groupby('sentiment').sample(n=samples, random_state=42)
print(dataset.head())
c, u  = np.unique(dataset['sentiment'], return_counts=True)
print(dict(zip(c, u)))

#Preprocessing
def preprocess_sentence(sentence):
    sentence = re.sub("[A-Z]([a-z]+|\.)(?:\s+[A-Z]([a-z]+|\.))*(?:\s+[a-z][a-z\-]+){0,2}\s+[A-Z]([a-z]+|\.)", " ", sentence)
    ret = sentence.lower()
    ret = re.sub("([¡¿:;\-_?.!,])", " ", ret)
    ret = re.sub("[\"\"]", " ", ret)
    ret = re.sub("@\S+|https?://\S+", " ", ret)
    # Delete hastags ---> model that learns the sentiment of a hastag
    ret = re.sub("\#\w+", "", ret)
    return ret

#def preprocess_sentence(sentence):
#    ret = sentence.lower()
#    ret = ret.strip()
#    ret = re.sub("([?.!,])", " \1 ", ret)
#    ret = re.sub('[" "]+', " ", ret)
#    ret = re.sub("a-zA-Z?.!,]+", " ", ret)
#    ret = ret.strip()
#    return ret

dataset['content'] = dataset['content'].map(lambda x: preprocess_sentence(x))


class Embeddings():
    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, tokenizer, max_features):
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix

X_data, y_data = dataset.drop('sentiment', axis=1), dataset['sentiment'].copy()

# Sentences to tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data['content'])

vocab_size = len(tokenizer.word_counts)
print('Vocab size: ', vocab_size)

#Word embedding
embedding = Embeddings(
    path = 'glove.6B.100d.txt',
    vector_dimension = 100,
)
embedding_matrix = embedding.create_embedding_matrix(tokenizer, vocab_size)
embedding_dim = 100

max_len = max([len(el) for el in dataset['content']])
average_len = int(sum(map(len, dataset['content']))/float(len(dataset['content'])))
print('Max len: ', max_len, 'Average len: ', average_len)

X_data = tokenizer.texts_to_sequences(X_data['content'])
X_data = pad_sequences(X_data, maxlen=75)
max_len = 75 # For using max_len as model hyperparameters

# One hot encode labels
y_data = np.eye(len(y_data.unique()))[y_data]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1)

# import pickle

# # Because we are training with the same data, doesn't make any sense to save the tokenizer every time
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

# Model
class RnnModel():
    def __init__(self, embedding_matrix, embedding_dim, max_len):
        inp1 = tf.keras.Input(shape=(max_len,))
        x = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = tf.keras.layers.LSTM(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x) 
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(3, activation="softmax")(x)    

        model = tf.keras.Model(inputs=inp1, outputs=x)

        opt = tf.keras.optimizers.Adam()

        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.model = model

# One hot encode with the vocabulary 
rnn = RnnModel(embedding_matrix, embedding_dim, max_len)
print(rnn.model.summary())

rnn.model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

rnn.model.save('model_v6')

print(X_test.shape)
i = 15999
ii = i + 1 
pred = rnn.model.predict(X_test[i:ii])[0]

print(tokenizer.sequences_to_texts(X_test[i:ii]), np.argmax(y_test[i:ii][0]), np.argmax(pred))
print(pred)

ips = ["today i read a book and eat some fish", "last week i ate a salad"]

nip = [[],[]]
for ii, ip in enumerate(ips):
    for i, word in enumerate(ip.split()):
        nip[ii].append(tokenizer.word_index[word])

print(nip)
text = pad_sequences(np.array(nip), 161)

pred = rnn.model.predict(text)
print('1º: ', pred[0], max([pp for pp in pred[0]]))
print('2º: ', pred[1], max([pp for pp in pred[1]]))
'''
!zip model_v5.zip model_v4/*

!zip model_v5.zip model_v4/*/*

bar = pd.read_csv('run/logger.csv')
bar
'''
# from google.colab import drive
# drive.mount('/content/gdrive',force_remount=True)

# !cp model_v4.zip '/content/gdrive/My Drive/downloads/'
# !ls -lt '/content/gdrive/My Drive/downloads/'
