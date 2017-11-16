
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import sklearn
import nltk
import tensorflow as tf
import matplotlib
import gzip
import random
from collections import defaultdict
import sklearn.decomposition


def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


# In[2]:


data = []
for l in readGz('train.json.gz'):
    data.append(l)


# In[3]:


userList = []
businessList = []
userDict = defaultdict(int)
businessDict = defaultdict(int)
visitedDict = defaultdict(int)
for d in data:
    u = d['userID']
    b = d['businessID']
    if u not in userList:
        userList.append(u)
    if b not in businessList:
        businessList.append(b)
    visitedDict[(u,b)]+=1

    
for i in range(len(userList)):
    userDict[userList[i]] = i
for i in range(len(businessList)):
    businessDict[businessList[i]] = i


# In[4]:


random.shuffle(data)
train_data = data[:160000]
validation_data = data[160000:200000]
negative_pair = []
randomCnt = 0
while(randomCnt) < 120000:
    ruIndex = random.randint(0, len(userList)-1)
    rbIndex = random.randint(0, len(businessList)-1)
    ru = userList[ruIndex]
    rb = businessList[rbIndex]
    if visitedDict[(ru, rb)]==0:
        randomCnt +=1
        negative_pair.append((ru,rb))
#Finished creating dataset

ratings = np.array([d['rating'] for d in train_data])
avgRating = np.mean(ratings)
print avgRating


# In[5]:


def Mean(lst):
    if len(lst) != 0:
        return np.mean(lst)
    else:
        return avgRating


# In[7]:


businessRat = [[] for i in range(len(businessList))]
userRat = [[] for i in range(len(userList))]
for d in train_data:
    businessRat[businessDict[d['businessID']]].append(d['rating'])
    userRat[userDict[d['userID']]].append(d['rating'])
businessAvg = [Mean(businessRat[i])-avgRating for i in range(len(businessList))]
userAvg = [Mean(l) for l in userRat]
print "Finished computing mean"


# In[8]:

popularList = [0 for i in range(len(businessList))]
for d in train_data:
    popularList[businessDict[d['businessID']]] += 1.0
popularList = np.array(popularList)

categoryList = []
for d in data:
    for c in d['categories']:
        if c not in categoryList:
            categoryList.append(c)
categoryDict = defaultdict(int)
for c in categoryList:
    categoryDict[c] = categoryList.index(c)

categoryPopular = [0 for i in range(len(categoryList))]


businessCategory = [[0 for j in range(len(categoryList))] for i in range(len(businessList))]
userHistory = [[0 for j in range(len(categoryList))] for i in range(len(userList))]
userCatAvg = [[[] for j in range(len(categoryList))]for i in range(len(userList))]
for d in train_data:
    for c in d['categories']:
        userHistory[userDict[d['userID']]][categoryDict[c]] += 1.0
        businessCategory[businessDict[d['businessID']]][categoryDict[c]] = 1.0
        userCatAvg[userDict[d['userID']]][categoryDict[c]].append(d['rating'])
        categoryPopular[categoryDict[c]] += 1.
userCatAvg_ = [[Mean(c)-userAvg[i] for c in userCatAvg[i]] for i in range(len(userList))]
meanCategory = np.mean(businessCategory, axis = 0)
#meanCategory = np.divide(meanCategory, np.linalg.norm(meanCategory))
meanUser = np.mean(userHistory, axis = 0)
#meanUser = np.divide(meanUser, np.linalg.norm(meanUser))
categoryPopular = np.array(categoryPopular)
normalized_categoryPopular = np.divide(categoryPopular, np.linalg.norm(categoryPopular))
#the list of what each business's category is and what category each user has visited


# In[5]:


def feature(datum):
    feat = []
    userOneHot = [0 for u in userList]
    businessOneHot = [0 for b in businessList]
    userOneHot[userDict[datum['userID']]] = 1
    businessOneHot[businessDict[datum['businessID']]] = 1
    feat.append(np.array(userOneHot))
    feat.append(np.array(businessOneHot))
    return feat


# In[6]:


#extracting features and ratings

train_rating = [d['rating'] for d in train_data]

validation_rating = [d['rating'] for d in validation_data]


# In[7]:


#tensorflow learning hyper parameters
learning_rate = 0.001
regularization_rate = 0.01
Reduction_Size = 100
max_iter = 2500
batch_size = 150


# In[26]:


def calc(user, business, regularizer, betau, betai, gammau, gammai):
    predict = avgRating

        
    predict += tf.matmul(user, betau)
    tf.add_to_collection('losses', regularizer(betau))
        
    predict += tf.matmul(business, betai)
    tf.add_to_collection('losses', regularizer(betai))
     
    ui = tf.matmul(tf.matmul(user, gammau), tf.matmul(gammai, tf.transpose(business)))
    #print tf.shape(ui)
    ui = tf.diag_part(ui)
    ui = tf.reshape(ui, [batch_size,1])
    predict = predict + ui
    tf.add_to_collection('losses', regularizer(gammau))
    tf.add_to_collection('losses', regularizer(gammai))
    
    return predict


def train():
    betai = tf.get_variable(name = 'betai', shape = [len(businessList),1], initializer = tf.truncated_normal_initializer(stddev = 0.5))
    betau = tf.get_variable(name = 'betau', shape = [len(userList),1], initializer = tf.truncated_normal_initializer(stddev = 0.5))
    gammau = tf.get_variable(name = 'gu', shape = [len(userList), Reduction_Size], initializer = tf.truncated_normal_initializer(stddev = 0.5))
    gammai = tf.get_variable(name = 'gi', shape = [Reduction_Size, len(businessList)], initializer = tf.truncated_normal_initializer(stddev = 0.5))
    U = tf.placeholder(tf.float32, [None, len(userList)], name = 'input-u')
    B = tf.placeholder(tf.float32, [None, len(businessList)], name = 'input-b')
    rating = tf.placeholder(tf.float32, [None], name = 'rating')
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    predict_ = calc(U,B,regularizer,betau, betai, gammau,gammai)
    predict_ = tf.reshape(predict_, [batch_size])
    se = tf.losses.mean_squared_error(labels = rating, predictions = predict_)
    sum_loss = tf.reduce_mean(se)
    loss = sum_loss + tf.add_n(tf.get_collection('losses'))
    
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(max_iter):
            print i
            sample = random.sample(zip(train_data,train_rating), batch_size)
            x_batch, y_batch = zip(*sample)
            x_batch_feature = [feature(d) for d in x_batch]
            x_batch_user = np.array([b[0] for b in x_batch_feature])
            x_batch_business = np.array([b[1] for b in x_batch_feature])
            
            l_value, _ = sess.run([loss, train_op], feed_dict = {U:x_batch_user, B: x_batch_business, rating:y_batch})
            if i % 10 == 0:
                print("After %d iters, loss on training batch is %f."%(i, l_value))
                sample_ = random.sample(zip(validation_data, validation_rating), batch_size)
                v_batch, v_rating = zip(*sample_)
                v_batch_feature = [feature(d) for d in v_batch]
                v_batch_user = np.array([v[0] for v in v_batch_feature])
                v_batch_business = np.array([v[1] for v in v_batch_feature])
                vl_value = sess.run(loss, feed_dict = {U:v_batch_user, B: v_batch_business, rating : v_rating})
                print("After %d iters, loss on validation batch is %f"%(i, vl_value))
        #betau_, betai_, gammau_,gammai_ = sess.run([betau, betai, gammau, gammai])
        betau_ = betau.eval()
        betai_ = betai.eval()
        gammau_ = gammau.eval()
        gammai_ = gammai.eval()
        return betau_, betai_, gammau_, gammai_

# In[28]:


bu, bi, gu, gi = train()
bu = np.array(bu)
#print gu[0]
#print gu[1]
def feat(u_,b_):
    u = userDict[u_]
    b = businessDict[b_]
    #print type(u)

    feature = [businessAvg[b]-avgRating if businessAvg[b]!=0 else 0]
    feature.append(popularList[b] if popularList[b]!=0 else 160000.0/len(businessList))

    feature = np.array(feature)
    cat = np.array(businessCategory[b])
    if np.linalg.norm(cat) == 0:
        cat = meanCategory
    feature = np.append(feature, np.inner(cat, normalized_categoryPopular))


    uHis = userHistory[u]
    if np.linalg.norm(uHis) == 0:
        uHis = meanUser
    feature = np.concatenate((feature, gu[u]))
    feature = np.concatenate((feature, gi.T[b]))
    feature = np.concatenate((feature, np.multiply(gu[u], gi.T[b])))
    
    return feature

'''train_feature = np.array([feat(d['userID'], d['businessID']) for d in train_data].extend([feat(u,b) for u,b in negative_pair[:60000]]))
validation_feature = np.array([feat(d['userID'], d['businessID']) for d in validation_data].extend([feat(u,b) for u,b in negative_pair[60000:100000]]))
train_label = np.array([1 for i in range(160000)].extend([0 for i in range(60000)]))
validation_feature = np.array([1 for i in range(40000)].extend([0 for i in range(40000)]))'''
print "Extracted feature after getting matrix compression"
train_feature = [feat(d['userID'], d['businessID']) for d in train_data]
train_feature.extend([feat(u,b) for u,b in negative_pair[:60000]])
validation_feature = [feat(d['userID'], d['businessID']) for d in validation_data]
validation_feature.extend([feat(u,b) for u,b in negative_pair[60000:100000]])
train_label = [1 for i in range(160000)]
train_label.extend([0 for i in range(60000)])
validation_label = [1 for i in range(40000)]
validation_label.extend([0 for i in range(40000)])
train_feature = np.array(train_feature)
validation_feature = np.array(validation_feature)
train_label = np.array(train_label)
validation_label = np.array(validation_label)
print np.shape(train_feature)
print np.shape(validation_feature)
print np.shape(train_label)
print np.shape(validation_label)

input_size = 303
fc_size = 60
output_size = 2
regularize_rate = 0.01
learn_rate = 0.001
iteration = 10000

def inference(X, regularizer):
    with tf.variable_scope('fc1'):
        w1 = tf.get_variable(name = 'weight', shape = [input_size, fc_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b1 = tf.get_variable(name = 'bias', shape = [fc_size], initializer = tf.constant_initializer(0.1))
        tf.add_to_collection('losses', regularizer(w1))
        fc1 = tf.nn.relu(tf.matmul(X, w1)+b1)

    with tf.variable_scope('fc2'):
        w2 = tf.get_variable(name = 'weight', shape = [fc_size, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.2))
        b2 = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.1))
        tf.add_to_collection('losses', regularizer(w2))
        fc2 = tf.matmul(fc1, w2)+b2

    return fc2

def train_nn():
    X = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.int32, [None])
    regularizer = tf.contrib.layers.l2_regularizer(regularize_rate)

    y_ = inference(X, regularizer)
    correct = tf.cast(tf.equal(y_,y), tf.float32)
    accuracy = tf.reduce_mean(correct)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_, labels = y))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(iteration):
            sample = np.random.randint(0, 220000, batch_size)
            x_batch = train_feature[sample]
            y_batch = train_label[sample]

            _, loss_value = sess.run([train_op, loss], feed_dict = {X:x_batch, y:y_batch})
            if i%500 == 0:
                print("After %d iterations, loss on training batch is %f"%(i, loss_value))
                acc = sess.run(accuracy, feed_dict = {X:validation_feature, y:validation_label})
                print("After %d iterations, accuracy on validation set is %f"%(i, acc))

train_nn()
