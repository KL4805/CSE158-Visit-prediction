
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import sklearn
from collections import defaultdict
import string
import gzip
from nltk.corpus import stopwords
import random
import math
import tensorflow as tf


# In[2]:


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)


# In[3]:


data = [l for l in readGz('train.json.gz')if 'categoryID' in l]


# In[4]:


print len(data)


# In[5]:


punctuation = set(string.punctuation)
stemmer = nltk.stem.porter.PorterStemmer()
#first try use stemmer
stopword = stopwords.words('english')


# In[6]:


wordCount = defaultdict(int)
for d in data:
    review = ''.join(c for c in d['reviewText'].lower() if c not in punctuation)
    wordList = review.split()
    for w in wordList:
        #w = stemmer.stem(w)
        if w not in stopword:
            wordCount[w] += 1
#first try not using stemmer


# In[7]:


print len(wordCount)


# In[8]:


count = [(wordCount[w], w ) for w in wordCount]
count.sort()
count.reverse()

commonWords = [count[i][1] for i in range(1000)]
wordDict = defaultdict(int)
for i in range(1000):
    wordDict[commonWords[i]] = i

#The 1000 most common words


# In[9]:


random.shuffle(data)
train_label = np.array([d['categoryID'] for d in data[:58000]])
train_data = data[:58000]
validation_label = np.array([d['categoryID'] for d in data[58000:70195]])
validation_data = data[58000:70195]


# In[11]:


#calculate tf-idf
#tf can be calculated when extracting feature
#idf calculated here
idf = [0 for i in range(1000)]
for d in train_data:
    review = ''.join(c for c in d['reviewText'].lower() if c not in punctuation)
    wordList = review.split()
    for w in commonWords:
        if w in wordList:
            idf[wordDict[w]] += 1.0
            
idf = np.array([math.log(70195.0/f) for f in idf])


# In[13]:

def feature(datum):
    #count tf-idf
    review = ''.join(c for c in d['reviewText'].lower() if c not in punctuation)
    wordList = review.split()
    tf = [0 for i in range(1000)]
    for w in wordList:
        if w in commonWords:
            tf[wordDict[w]] += 1.0
    tf = np.array(tf)
    if np.max(tf) != 0:
        tf_ = np.divide(tf, np.max(tf))
    else:
        tf_ = tf
    tfidf = np.multiply(tf_,idf)
    return tfidf


# In[14]:


train_feature = np.array([feature(d) for d in train_data])
validation_feature = np.array([feature(d) for d in validation_data])
test_data = []
for l in readGz("test_Category.json.gz"):
    test_data.append(l)
test_feature = np.array([feature(d) for d in test_data])

# In[15]:


fc_size = 400
input_size = 1000
output_size = 10
regularization_rate = 0.00001
learning_rate = 0.00005
batch_size = 256
max_iter = 10000
#tensorflow learning hyperpatameters


# In[ ]:


def calc(X, regularizer):
    with tf.variable_scope('fc1'):
        w1 = tf.get_variable(name = 'weight', shape = [input_size, fc_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b1 = tf.get_variable(name = 'bias', shape = [fc_size], initializer = tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(X, w1)+b1)
        tf.add_to_collection('losses', regularizer(w1))
    

    with tf.variable_scope('fc2'):
        w2 = tf.get_variable(name = 'weight', shape = [fc_size, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b2 = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, w2) + b2
        tf.add_to_collection('losses', regularizer(w2))
    
    return fc2
#A neural network with one hidden layer


# In[16]:


def train():
    X = tf.placeholder(tf.float32, [None, input_size], name = 'input-X')
    y = tf.placeholder(tf.int64, [None], name = 'input-Y')
    
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    
    y_ = calc(X, regularizer)
    y_predict = tf.argmax(y_,1)
    correct_prediction = tf.cast(tf.equal(y_predict, y),tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_, labels = y))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(max_iter):
            sample = np.random.randint(0, 58000, batch_size)
            #print sample
            x_batch = train_feature[sample]
            y_batch = train_label[sample]
            
            _, loss_value = sess.run([train_step, loss], feed_dict = {X:x_batch,y:y_batch})
            if i % 1000 == 0:
                print("After %d iters, loss on training is %f."%(i, loss_value))
                acc = sess.run(accuracy, feed_dict = {X:validation_feature, y:validation_label})
                print("After %d iters, accuracy on validation is %f"%(i, acc))

        print "Training finish"
        predictions = open("predictions_Category.txt", 'w')
        predictions.write("userID-reviewHash,category\n")
        y_p = sess.run(y_predict, feed_dict = {X : test_feature})
        for d, l in zip(test_data, y_p):
            predictions.write(d['userID'] + '-' + d['reviewHash'] + ',' + str(l) + '\n')
# In[ ]:


train()

