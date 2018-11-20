#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, BatchNormalization, Activation
from keras.layers.core import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import dot
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import regularizers
import jieba
from keras.initializers import Constant



X_train1 = []

X_train2 = []

y_train = []
with open('data/simtrain_to05sts.txt',encoding = 'utf-8') as f:
    for line in f.readlines():
        text = line.strip().split('\t')

        X_train1.append(' '.join(jieba.cut(text[1])))

        X_train2.append(' '.join(jieba.cut(text[3])))

        y_train.append(float(text[-1]))

y_train_1 = [ 0 if i<=2 else 1 for i in y_train]

print('Indexing word vectors.')

embeddings_index = {}

with open('pre_train_words/sgns.zhihu.bigram',encoding = 'utf-8') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs


print('Found %s word vectors.' % len(embeddings_index))


MAX_NUM_WORDS = 20000

MAX_SEQUENCE_LENGTH = 50

all_text = X_train1+X_train2

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(all_text)

sequences = tokenizer.texts_to_sequences(all_text)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

X1 = data[:len(X_train1)]
X2 = data[len(X_train2):]



print('Preparing embedding matrix.')

EMBEDDING_DIM = 300


num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():

    if i > MAX_NUM_WORDS:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)


# In[61]:


inx = [i for i in range(len(X1))]

np.random.shuffle(inx)

X1 = np.array(X1)[inx]

X2 = np.array(X2)[inx]

y_train_1 = np.array(y_train_1)[inx]




print('Build model...')

num_conv2d_layers=2

filters_2d=[16,32]

kernel_size_2d=[[3,3], [3,3]]

mpool_size_2d=[[2,2], [2,2]]

dropout_rate=0.5

batch_size=128

query=Input(shape=(MAX_SEQUENCE_LENGTH,), name='query')

doc=Input(shape=(MAX_SEQUENCE_LENGTH,), name='doc')

q_embed=embedding_layer(query)

d_embed=embedding_layer(doc)

layer1_dot=dot([q_embed, d_embed], axes=-1)

layer1_dot=Reshape((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, -1))(layer1_dot)
    
layer1_conv=Conv2D(filters=8, kernel_size=5, padding='same')(layer1_dot)

layer1_activation=Activation('relu')(layer1_conv)

z=MaxPooling2D(pool_size=(2,2))(layer1_activation)
    
for i in range(num_conv2d_layers):
    
    z=Conv2D(filters=filters_2d[i], kernel_size=kernel_size_2d[i], padding='same')(z)
    
    z=Activation('relu')(z)
    
    z=MaxPooling2D(pool_size=(mpool_size_2d[i][0], mpool_size_2d[i][1]))(z)
        
pool1_flat=Flatten()(z)

pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)

mlp1=Dense(128)(pool1_flat_drop)

mlp1=Activation('relu')(mlp1)

out=Dense(1, activation='sigmoid')(mlp1)
    
model=Model(inputs=[query, doc], outputs=out)

model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['acc'])

model.fit([X1,X2],y_train_1,epochs = 20,batch_size = batch_size,validation_split = 0.2,class_weight={0:0.75,1:0.25})


#model.summary()




