
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
    wordList = [stemmer.stem(w) for w in wordList]
    d['wordList'] = wordList
    for w in wordList:
        if w not in stopword:
            wordCount[w] += 1
#first try not using stemmer


# In[7]:


print len(wordCount)


# In[8]:


count = [(wordCount[w], w ) for w in wordCount]
count.sort()
count.reverse()

commonWords = [count[i][1] for i in range(3000)]
wordDict = defaultdict(int)
for i in range(2000):
    wordDict[commonWords[i]] = i

#The 1000 most common words


# In[9]:


random.shuffle(data)
train_label = np.array([d['categoryID'] for d in data[:60000]])
train_data = data[:60000]
validation_label = np.array([d['categoryID'] for d in data[60000:70195]])
validation_data = data[60000:70195]


# In[11]:

def Mean(lst):
    if len(lst) != 0:
        return np.mean(lst)
    else:
        return 
userList = []
for d in train_data:
    if d['userID'] not in userList:
        userList.append(d['userID'])
print len(userList)
userDict = defaultdict(int)
userAvg = [[] for u in userList]
for i in range(len(userList)):
    userDict[userList[i]] = i
userHistory = [[0 for i in range(10)] for j in range(len(userList))]
for d in train_data:
    userHistory[userDict[d['userID']]][d['categoryID']] += 1.0
    userAvg[userDict[d['userID']]].append(d['rating'])
userAvg = [np.mean(l) for l in userAvg]
userCatRating = [[[] for i in range(10)] for u in userList]
for d in train_data:
    userCatRating[userDict[d['userID']]][d['categoryID']].append(d['rating'])
userCatAvg = [[np.mean(userCatRating[u][i]) if len(userCatRating[u][i])!= 0 else userAvg[u] for i in range(10)] for u in range(len(userList))]
print userCatAvg[:10]
userHistory = np.array(userHistory)
print np.shape(userAvg)
#calculate tf-idf
#tf can be calculated when extracting feature
#idf calculated here
idf = [0 for i in range(2000)]
for d in train_data:

    for w in d['wordList']:
        if w in commonWords:
            idf[wordDict[w]] += 1.0
            
idf = np.array([math.log(60000.0/f) for f in idf])


meanUser = np.mean(userHistory, 0)
# In[13]:

def feature(datum):
    #count tf-idf
    tf = [0 for i in range(2000)]
    for w in datum['wordList']:
        if w in commonWords:
            tf[wordDict[w]] += 1.0
    tf = np.array(tf)
    if np.max(tf) != 0:
        tf_ = np.divide(tf, np.max(tf))
    else:
        tf_ = tf
    tfidf = np.multiply(tf_,idf)
    if datum['userID'] in userList:
        tfidf = np.concatenate((tfidf, userHistory[userDict[datum['userID']]]))
    else:
        tfidf = np.concatenate((tfidf, meanUser))
    if datum['userID'] in userList:
        tfidf = np.concatenate((tfidf, userCatAvg[userDict[datum['userID']]]))
    else:
        tfidf = np.concatenate((tfidf, np.array([np.mean(userAvg) for i in range(10)])))
    return tfidf


# In[14]:


train_feature = np.array([feature(d) for d in train_data])
validation_feature = np.array([feature(d) for d in validation_data])
test_data = []
for l in readGz("test_Category.json.gz"):
    test_data.append(l)
for d in test_data:

    review = ''.join(c for c in d['reviewText'].lower() if c not in punctuation)
    wordList = review.split()
    wordList = [stemmer.stem(w) for w in wordList]
    d['wordList'] = wordList
test_feature = np.array([feature(d) for d in test_data])

# In[15]:


fc_size = 500
fc2_size = 50
fc3_size = 70
input_size = 2000
output_size = 10
regularization_rate = 0.007
learning_rate = 0.0001
batch_size = 200
max_iter = 30000
#tensorflow learning hyperpatameters


# In[ ]:


def calc(X, u, regularizer, dropout):
    with tf.variable_scope('fc1'):
        w1 = tf.get_variable(name = 'weight', shape = [input_size, fc_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b1 = tf.get_variable(name = 'bias', shape = [fc_size], initializer = tf.constant_initializer(0.1))       
        #X_NN = tf.slice(X, [0,0], [tf.shape(X)[0], 1500])
        fc1 = tf.nn.relu(tf.matmul(X, w1)+b1)
        dropouted = tf.layers.dropout(fc1, dropout) 
        tf.add_to_collection('losses', regularizer(w1))
    

    with tf.variable_scope('fc2'):
        w2 = tf.get_variable(name = 'weight', shape = [fc_size, fc2_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b2 = tf.get_variable(name = 'bias', shape = [fc2_size], initializer = tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)
        #fc2 = tf.matmul(dropouted, w2)+b2
        tf.add_to_collection('losses', regularizer(w2))

    in3 = tf.concat([fc2,u], 1)

    with tf.variable_scope('fc3'):
        w3 = tf.get_variable(name = 'weight', shape = [fc3_size, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b3 = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.1))
        fc3 = tf.matmul(in3, w3)+b3

    return fc3

    '''with tf.variable_scope('log1'):
        w3 = tf.get_variable(name = 'weight', shape = [10, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b3 = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.1))
        X_log = tf.slice(X,[0,1500],[tf.shape(X)[0], 10])
        log1 = tf.matmul(X_log, w3)+b3'''
    

#A neural network with 2 hidden layer


# In[16]:


def train():
    X = tf.placeholder(tf.float32, [None, input_size], name = 'input-X')
    U = tf.placeholder(tf.float32, [None, 20], name = 'input-u')
    y = tf.placeholder(tf.int64, [None], name = 'input-Y')
    dropout_r = tf.placeholder(tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    
    y_ = calc(X,U, regularizer, dropout_r)
    y_predict = tf.argmax(y_,1)
    correct_prediction = tf.cast(tf.equal(y_predict, y),tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_, labels = y))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(max_iter):
            sample = np.random.randint(0, 60000, batch_size)
            #print sample
            x_batch = train_feature[sample, :2000]
            x_batch_u = train_feature[sample, 2000:2020]
            y_batch = train_label[sample]
            
            _, loss_value = sess.run([train_step, loss], feed_dict = {X:x_batch,U:x_batch_u,y:y_batch, dropout_r:0.5})
            if i % 1000 == 0:
                print("After %d iters, loss on training is %f."%(i, loss_value))
                acc = sess.run(accuracy, feed_dict = {X:validation_feature[:, :2000],U:validation_feature[:, 2000:2020],y:validation_label, dropout_r:1})
                print("After %d iters, accuracy on validation is %f"%(i, acc))

        print "Training finish"
        predictions = open("predictions_Category.txt", 'w')
        predictions.write("userID-reviewHash,category\n")
        y_p = sess.run(y_predict, feed_dict = {X : test_feature[:,:2000], U:test_feature[:, 2000:2020],dropout_r:1})
        for d, l in zip(test_data, y_p):
            predictions.write(d['userID'] + '-' + d['reviewHash'] + ',' + str(l) + '\n')
# In[ ]:


train()
#neural network method ends here

