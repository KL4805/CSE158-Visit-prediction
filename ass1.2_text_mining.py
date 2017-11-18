
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
from scipy.spatial.distance import cosine


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

    wordList = [stemmer.stem(w) for w in wordList if w not in stopword]
    d['wordList'] = wordList
    for w in wordList:
        wordCount[w] += 1
#use stemmer


# In[7]:


print len(wordCount)


# In[8]:


lenWords = 2000


# In[9]:


count = [(wordCount[w], w ) for w in wordCount]
count.sort()
count.reverse()

commonWords = [t[1] for t in count[:lenWords]]

wordDict = defaultdict(int)
for i in range(lenWords):
    wordDict[commonWords[i]] = i

#The 1000 most common words


# In[10]:


random.shuffle(data)
train_label = np.array([d['categoryID'] for d in data[:60000]])
train_data = data[:60000]
validation_label = np.array([d['categoryID'] for d in data[60000:70195]])
validation_data = data[60000:70195]


# In[11]:


#calculate idf
#tf calculated when creating feature
idf = [0 for i in range(lenWords)]

for d in train_data:
    reviewSet = set(d['wordList'])
    for w in commonWords:
        if w in reviewSet:
            idf[wordDict[w]] += 1.0


# In[12]:


idf = [math.log(60000/f) for f in idf]
idf = np.array(idf)


# In[13]:


avgRating = np.mean([d['rating'] for d in train_data])


# In[37]:


userList = []
businessList = []
for d in train_data:
    if d['userID'] not in userList:
        userList.append(d['userID'])
    if d['businessID'] not in businessList:
        businessList.append(d['businessID'])
userDict = defaultdict(int)
businessDict = defaultdict(int)
for i in range(len(userList)):
    userDict[userList[i]] = i
for i in range(len(businessList)):
    businessDict[businessList[i]] = i
userHistory = [[0 for i in range(10)] for j in range(len(userList))]
userCatRating = [[[] for i in range(10)]for u in userList]


# In[39]:


#building lists of user's visited businesses
u_visited = [defaultdict(float) for u in userList]
b_visited = [defaultdict(float) for b in businessList]
for d in train_data:
    u = userDict[d['userID']]
    b = businessDict[d['businessID']]
    rating = d['rating']
    u_visited[u][b] = rating
    b_visited[b][u] = rating


# In[41]:


#Get average rating of all businesses and all users
userAvg = [np.mean([u_visited[u][t] for t in u_visited[u]]) for u in range(len(userList))]
businessAvg = [np.mean([b_visited[b][t] for t in b_visited[b]]) for b in range(len(businessList))]
avgRating = np.mean([d['rating'] for d in train_data])


# In[42]:


def uJaccard(u1, u2):
    u1Set = set([b for b in u_visited[u1]])
    u2Set = set([b for b in u_visited[u2]])
    return (len(u1Set & u2Set)*1.0)/len(u1Set | u2Set)


# In[43]:


def uPearson(u1,u2):
    u1Set = set([b for b in u_visited[u1]])
    u2Set = set([b for b in u_visited[u2]])
    u1rList = []
    u2rList = []
    bavg = []
    for b in (u1Set & u2Set):
        u1rList.append(u_visited[u1][b])
        u2rList.append(u_visited[u2][b])
        bavg.append(businessAvg[b])

    if len(u1Set & u2Set) != 0:
        cov = np.sum([(u1rList[i]-bavg[i])*(u2rList[i]-bavg[i]) for i in range(len(u1rList))])
        std = math.sqrt(np.sum([(r-a)**2 for r,a in zip(u1rList, bavg)]) * np.sum(([(r-a)**2 for r,a in zip(u2rList, bavg)])))
        return (cov*1.0)/std if std != 0 else 0
    else:
        return 0


# In[31]:


for d in train_data:
    u = userDict[d['userID']]
    c = d['categoryID']
    userCatRating[u][c].append(d['rating'])
    userHistory[u][c]+=1.0
userCatAvg = [[np.mean(l)-userAvg[u] if len(l)!=0 else 0 for l in userCatRating[u] ]for u in range(len(userList))]


# In[16]:


userHistory = [np.divide(u, np.linalg.norm(u)) for u in userHistory]


# In[17]:


def feat(datum):
    wordList = datum['wordList']
    tf = [0 for i in range(lenWords)]
    for w in wordList:
        if w in commonWords:
            tf[wordDict[w]] += 1.0
    tf = np.array(tf)
    if np.max(tf) != 0 :
        tf = np.divide(tf, np.max(tf))
    tfidf = np.multiply(tf, idf)
    if datum['userID'] in userList:
        u = userDict[datum['userID']]
        tfidf = np.concatenate((tfidf, userHistory[u]))
        tfidf = np.concatenate((tfidf, userCatAvg[u]))
    else:
        tfidf = np.concatenate((tfidf, [0 for i in range(20)]))
    return tfidf


# In[18]:


train_feature = np.array([feat(d) for d in train_data])
validation_feature = np.array([feat(d) for d in validation_data])


# In[19]:


print np.shape(train_feature)


# In[20]:


avg_tfidf = [[]for i in range(10)]
for f, l in zip(train_feature, train_label):
    avg_tfidf[l].append(f[0:lenWords])
avg_tfidf = [np.mean(v, 0) for v in avg_tfidf]


# In[21]:


print np.shape(avg_tfidf)


# In[22]:


def feature(f):
    if np.linalg.norm(f[:2000]) !=0:
        f = np.concatenate((f, [cosine(f[:2000], avg_tfidf[i]) for i in range(10)]))
    else:
        f = np.concatenate((f, [0 for i in range(10)]))
    return f


# In[23]:


train_feature = np.array([feature(d) for d in train_feature])


# In[24]:


print np.shape(train_feature)


# In[25]:


test_data = []
for l in readGz("test_Category.json.gz"):
    test_data.append(l)
for d in test_data:

    review = ''.join(c for c in d['reviewText'].lower() if c not in punctuation)
    wordList = review.split()
    wordList = [stemmer.stem(w) for w in wordList if w not in stopword]
    d['wordList'] = wordList
test_feature = np.array([feat(d) for d in test_data])
test_feature = np.array([feature(d) for d in test_feature])


# In[26]:


validation_feature = [feature(d) for d in validation_feature]


# In[27]:


fc_size = 500
input_size = 2030
output_size = 10
regularization_rate = 0.01
learning_rate = 0.0001
batch_size = 200
max_iter = 60000
#tensorflow learning hyperparameters


# In[28]:


def calc(X, regularizer):
    with tf.variable_scope('fc1'):
        w1 = tf.get_variable(name = 'weight', shape = [input_size, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.1))
        b1 = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(X, w1)+b1)
        #fc1 = tf.matmul(X, w1)+b1
        tf.add_to_collection('losses', regularizer(w1))
    
    '''with tf.variable_scope('fc2'):
        w2 = tf.get_variable(name = 'weight', shape = [fc_size, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.1))
        b2 = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, w2) + b2
        tf.add_to_collection('losses', regularizer(w2))'''
    
    return fc1
#A neural network with one hidden layer


# In[29]:


X = tf.placeholder(tf.float32, [None, input_size], name = 'input-X')
y = tf.placeholder(tf.int64, [None], name = 'input-Y')
    
regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    
y_ = calc(X, regularizer)
y_predict = tf.argmax(y_,1)
correct_prediction = tf.cast(tf.equal(y_predict, y),tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_, labels = y)
loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[30]:


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(max_iter):
        sample = np.random.randint(0, 60000, batch_size)
        x_batch = train_feature[sample]
        y_batch = train_label[sample]
            
        _, loss_value = sess.run([train_step, loss], feed_dict = {X:x_batch,y:y_batch})
        if i % 500 == 0:
            print("After %d iters, loss on training is %f."%(i, loss_value))
            acc = sess.run(accuracy, feed_dict = {X:validation_feature, y:validation_label})
            print("After %d iters, accuracy on validation is %f"%(i, acc))
    predictions = open("predictions_Category.txt", 'w')
    predictions.write("userID-reviewHash,category\n")
    y_p = sess.run(y_predict, feed_dict = {X : test_feature,dropout_r:1})
    for d, l in zip(test_data, y_p):
        predictions.write(d['userID'] + '-' + d['reviewHash'] + ',' + str(l) + '\n')

