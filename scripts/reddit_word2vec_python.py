
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import nltk.data

from gensim.models import word2vec

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

import pandas as pd
import numpy as np

import os
import re
import logging
import sqlite3
import time
import sys
import multiprocessing
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from itertools import cycle
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


nltk.download('punkt')


# In[4]:


sql_con = sqlite3.connect("/home/cbrom/workspace/2015-02.db")


# In[5]:


start = time.time()
sql_data = pd.read_sql("SELECT parent, comment from parent_reply", sql_con)
print('Total time: ' + str((time.time() - start)) + ' secs')


# In[6]:


total_rows = len(sql_data)
print(total_rows)


# In[7]:


sql_data.head()


# In[8]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[9]:


def clean_text(pairs, out_name):
    start = time.time()
    with open(out_name, 'a', encoding='utf8') as out_file:
        for pos in range(len(pairs)):
            parent = pairs.iloc[pos]['parent'];
            comment = pairs.iloc[pos]['comment']
            
            # remove newlines
            par_no_tabs = str(parent).replace('\t', ' ')
            com_no_tabs = str(comment).replace('\t', ' ')
            
            # normalize to alpha and dot (.)
            par_alpha = re.sub("[^a-zA-Z\.]", " ", par_no_tabs)
            com_alpha = re.sub("[^a-zA-Z\.]", " ", com_no_tabs)
            
            # change multi space to 1
            par_multi_space = re.sub(" +", " ", par_alpha)
            com_multi_space = re.sub(" +", " ", com_alpha)
            
            # stripe 
            par_strip = par_multi_space.strip()
            com_strip = com_multi_space.strip()
            
            # lowercase
            par_clean = par_strip.lower()
            com_clean = com_strip.lower()
            
            # tokenize
            par_sents = tokenizer.tokenize(par_clean)
            par_sents = [re.sub("[\.]", "", par_sent) for par_sent in par_sents]
            com_sents = tokenizer.tokenize(com_clean)
            com_sents = [re.sub("[\.]", "", com_sent) for com_sent in com_sents]
            
            if len(par_clean) > 0 and par_clean.count(' ') > 0:
                for par_sent in par_sents:
                    out_file.write("%s\n" % par_sent)
            
            if len(com_clean) > 0 and com_clean.count(' ') > 0:
                for com_sent in com_sents:
                    out_file.write("%s\n" % com_sent)
                
            if pos % 50000 == 0:
                total_time = time.time() - start
                sys.stdout.write(
                    'Completed ' + 
                    str(round(100 * (pos/total_rows), 2)) + 
                    '% - ' + str(pos) + 
                    ' rows in time ' + 
                    str(round(total_time / 60, 0)) + 
                    ' min & ' + 
                    str(round(total_time % 60, 2)) + 
                    ' secs\r')
        out_file.flush()
                


# In[10]:


start = time.time();
clean_comments = clean_text(sql_data, '/home/cbrom/workspace/full.txt')
print('Total time: ' + str((time.time() - start)) + ' secs')


# In[15]:


start = time.time();

#Set the logging format to get some basic updates.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 100;    # Dimensionality of the hidden layer representation
min_word_count = 40;   # Minimum word count to keep a word in the vocabulary
num_workers = multiprocessing.cpu_count();       # Number of threads to run in parallel set to total number of cpus.
context = 5          # Context window size (on each side)                                                       
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model. 
#The LineSentence object allows us to pass in a file name directly as input to Word2Vec,
#instead of having to read it into memory first.

print("Training model...")
model = word2vec.Word2Vec(word2vec.LineSentence('/home/cbrom/workspace/full.txt'), workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

# We don
# Save the model
model_name = "model_full_reddit";
model.save(model_name) #t plan on training the model any further, so calling 
# init_sims will make the model more memory efficient by normalizing the vectors in-place.
model.init_sims(replace=True);


print('Total time: ' + str((time.time() - start)) + ' secs')


# In[16]:


model = word2vec.Word2Vec.load('model_full_reddit');


# In[17]:


Z = model.wv.vectors;


# In[18]:


print(Z[0].shape)
Z[0]


# In[7]:


def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++');
    idx = kmeans_clustering.fit_predict(word_vectors);
    
    return kmeans_clustering.cluster_centers_, idx;


# In[8]:


start = time.time();
centers, clusters = clustering_on_wordvecs(Z, 50);
print('Total time: ' + str((time.time() - start)) + ' secs')


# In[9]:


start = time.time();
centroid_map = dict(zip(model.wv.index2word, clusters));
print('Total time: ' + str((time.time() - start)) + ' secs')


# In[10]:


def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs);

    #Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers];
    closest_words_idxs = [x[1] for x in closest_points];

    #Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {};
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i+1).zfill(2)] = [index2word[j] for j in closest_words_idxs[i][0]]

    #A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words);
    df.index = df.index+1

    return df;


# In[11]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[12]:


top_words = get_top_words(model.wv.index2word, 20, centers, Z);


# In[13]:


top_words


# In[14]:




def display_cloud(cluster_num, cmap):
    wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=cmap);
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num).zfill(2)]]))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('cluster_' + str(cluster_num), bbox_inches='tight')



# In[15]:


cmaps = cycle([
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])

for i in range(50):
    col = next(cmaps);
    display_cloud(i+1, col)


# In[16]:


def get_word_table(table, key, sim_key='similarity', show_sim = True):
    if show_sim == True:
        return pd.DataFrame(table, columns=[key, sim_key])
    else:
        return pd.DataFrame(table, columns=[key, sim_key])[key]


# In[24]:


get_word_table(model.wv.most_similar_cosmul(positive=['realist', 'woman'], negative=['man']), 'Analogy')


# In[44]:


model.wv.most_similar_cosmul('king')


# In[26]:


q = model['king'] - model['man'] + model['woman']
# q = model['king'] - model['man'] + model['woman']
get_word_table(model.wv.similar_by_vector(q), 'Analogy')


# In[27]:


model.wv.doesnt_match("apple microsoft samsung tesla".split())


# In[28]:


model.wv.doesnt_match("trump clinton sanders obama".split())


# In[29]:


model.wv.doesnt_match("joffrey cersei tywin lannister jon".split())


# In[30]:


model.wv.doesnt_match("daenerys rhaegar viserion aemon aegon jon targaryen".split())


# In[31]:


keys = ['musk', 'modi', 'hodor', 'martell', 'apple', 'neutrality', 'snowden', 'batman', 'hulk', 'warriors', 
        'falcons', 'pizza', ];
tables = [];
for key in keys:
    tables.append(get_word_table(model.wv.similar_by_word(key), key, show_sim=False))


# In[32]:


pd.concat(tables, axis=1)


# In[33]:


import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


# In[46]:


Z.shape


# In[47]:


# project part of vocab, Z.shape dimension
w2v_sha = np.zeros(Z.shape)


# In[57]:


with open("./projector/prefix_metadat.tsv", "w+") as file_metadata:
    for i, word in enumerate(model.wv.index2word[:Z.shape[0]]):
        w2v_sha[i] = model[word]
        file_metadata.write(word + '\n')
        


# In[58]:


# define model
sess = tf.InteractiveSession()


# In[59]:


with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v_sha, trainable=False, name="prefix_embedding")

tf.global_variables_initializer().run()
saver = tf.train.Saver()
writer = tf.summary.FileWriter("./projector", sess.graph)


# In[60]:


# adding to the projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'prefix_embedding:0'
embed.metadata_path = 'prefix_metadat.tsv'


# In[61]:


# specifiy width and height of a single thumbnail
projector.visualize_embeddings(writer, config)
saver.save(sess, './projector/prefix_model.ckpt', global_step=Z.shape[0])

