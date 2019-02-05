
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
print(tf.__version__)
tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
import gensim
import pandas as pd
import nltk
import unicodedata

import unicodedata
import re
import numpy as np
import os
import sys
import time
import logging
import sqlite3
import multiprocessing
from itertools import cycle
import warnings

warnings.filterwarnings('ignore')


# In[2]:


cleaned_data = pd.read_csv('cleaned_data.csv', nrows=150000)


# In[3]:


cleaned_data = cleaned_data[cleaned_data['cleaned_parent'].map(
lambda val: len(val.split(' ')) <= 100)]
cleaned_data = cleaned_data[cleaned_data['cleaned_comment'].map(
lambda val: len(val.split(' ')) <= 100)]


# In[4]:


len(cleaned_data)


# In[5]:


# load word embedding
model = word2vec.Word2Vec.load('model_full_reddit');


# In[6]:


class DataWraper():
    def __init__(self, inp_data):
        self.inp_data = inp_data
        self.word2emb = lambda inp: model[inp]
        self.emb2word = lambda inp: model.similar_by_vector(a, topn=1)[0][0]
        self.tensor = [[model.wv.vocab[word].index for word in sent.split(' ') 
                       if word in model.wv.vocab]
                       for sent in self.inp_data]
#         self.word_tensor = [[word for word in sent.split(' ')]
#                            for sent in self.inp_data]
        self.max_length = max(len(sent.split(' ')) for sent in self.inp_data)


# In[7]:


def load_dataset(dataframe):
    parent_wraper = DataWraper(dataframe['cleaned_parent'])
    comment_wraper = DataWraper(dataframe['cleaned_comment'])
    parent_wraper.tensor = tf.keras.preprocessing.sequence.pad_sequences(
    parent_wraper.tensor, maxlen=parent_wraper.max_length,
    padding='post')
    
    comment_wraper.tensor = tf.keras.preprocessing.sequence.pad_sequences(
    comment_wraper.tensor, maxlen=comment_wraper.max_length,
    padding='post')
    
    return parent_wraper.tensor, comment_wraper.tensor, parent_wraper, comment_wraper, parent_wraper.max_length, comment_wraper.max_length


# In[8]:


input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(cleaned_data)


# In[9]:


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
print(max_length_targ)


# In[10]:


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 1
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 100
units = 1024

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# In[11]:


def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')


# In[12]:


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# In[13]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


# In[14]:


encoder = Encoder(len(model.wv.vocab),embedding_dim, units, BATCH_SIZE)
decoder = Decoder(len(model.wv.vocab),embedding_dim, units, BATCH_SIZE)


# In[15]:


optimizer = tf.train.AdamOptimizer()


def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)


# In[16]:


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# In[ ]:


EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()
    
    hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    for (batch, (inp, targ)) in enumerate(dataset):
        loss = 0
        
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)
            
            dec_hidden = enc_hidden
            
            dec_input = tf.expand_dims([14000] * BATCH_SIZE, 1)       
            
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                
                loss += loss_function(targ[:, t], predictions)
                
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        
        batch_loss = (loss / int(targ.shape[1]))
        
        total_loss += batch_loss
        
        variables = encoder.variables + decoder.variables
        
        gradients = tape.gradient(loss, variables)
        
        optimizer.apply_gradients(zip(gradients, variables))
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

