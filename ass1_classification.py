# coding: utf-8

# In[12]:


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

regularization_rate = 0.025

batch_size = 300
max_iter = 8000
learning_rate = 0.001


def readGz(f):
    for l in gzip.open(f):
        yield eval(l)




# In[13]:


data = []
for l in readGz('train.json.gz'):
    data.append(l)

test_data = []
predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
    if l.startswith('userID'):
        continue
    else:
        u,b = l.strip().split('-')
        test_data.append((u,b))



# In[14]:


userList = []
businessList = []
userDict = defaultdict(int)
businessDict = defaultdict(int)
visitedDict = defaultdict(int)
# random.shuffle(data)

for d in data:
    u = d['userID']
    b = d['businessID']
    if u not in userList:
        userList.append(u)
    if b not in businessList:
        businessList.append(b)
    visitedDict[(u, b)] += 1

for i in range(len(userList)):
    userDict[userList[i]] = i
for i in range(len(businessList)):
    businessDict[businessList[i]] = i

# In[15]:


random.shuffle(data)
train_data = data[0:140000]
validation_data = data[140000:200000]
negative_pair = []
randomCnt = 0
while (randomCnt) < 100000:
    ruIndex = random.randint(0, len(userList) - 1)
    rbIndex = random.randint(0, len(businessList) - 1)
    ru = userList[ruIndex]
    rb = businessList[rbIndex]
    if visitedDict[(ru, rb)] == 0:
        randomCnt += 1
        negative_pair.append((ru, rb))
# Finished creating dataset
# print negative_pair[:10]

ratings = np.array([d['rating'] for d in train_data])
avgRating = np.mean(ratings)
print avgRating


# In[16]:


def Mean(lst):
    if len(lst) != 0:
        return np.mean(lst)
    else:
        return avgRating


# In[17]:


businessRat = [[] for i in range(len(businessList))]
userRat = [[] for i in range(len(userList))]
for d in train_data:
    businessRat[businessDict[d['businessID']]].append(d['rating'])
    userRat[userDict[d['userID']]].append(d['rating'])
businessAvg = [Mean(businessRat[i])-avgRating for i in range(len(businessList))]
userAvg = [Mean(l) for l in userRat]
print "Finished computing mean"

# In[18]:

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
print len(categoryList)


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
meanCategory = np.divide(meanCategory, np.linalg.norm(meanCategory))
meanUser = np.mean(userHistory, axis = 0)
meanUser = np.divide(meanUser, np.linalg.norm(meanUser))
categoryPopular = np.array(categoryPopular)
normalized_categoryPopular = np.divide(categoryPopular, np.linalg.norm(categoryPopular))


'''uIs0 = [np.linalg.norm(u)==0 for u in userHistory]
print np.sum(uIs0)
bIs0 = [np.linalg.norm(b)==0 for b in businessCategory]
print np.sum(bIs0)'''
# In[32]:





# In[33]:


# upca = sklearn.decomposition.PCA(n_components = 100)
# userHistory = upca.fit_transform(userHistory)
# print np.isnan(userHistory).any()
# print np.isfinite(userHistory).all()
# print len(userHistory[0])


# In[31]:



# pca = sklearn.decomposition.PCA(n_components = 100)
# categoryFeature = pca.fit_transform(categoryFeature)
# print np.isnan(categoryFeature).any()
# print np.isfinite(categoryFeature).all()
# print len(categoryFeature[0])
# print "Finished doing PCA"
# In[39]:




def feature(b, u):
    feat = np.array([businessAvg[businessDict[b]], popularList[businessDict[b]]])
    
    cat = np.array(businessCategory[businessDict[b]])
    if np.linalg.norm(cat) == 0:
        cat_ = meanCategory
    else:
        cat_ = np.divide(cat, np.linalg.norm(cat))
    
    feat = np.append(feat, np.inner(cat_, normalized_categoryPopular))

    extension = np.array(userHistory[userDict[u]])
    if np.linalg.norm(extension) == 0:
        extension_ = meanUser
    else:
        extension_ = np.divide(extension, np.linalg.norm(extension))

    extent = np.inner(extension_, cat_)
    feat = np.append(feat, extent)
    extent2 = np.inner(cat_, np.array(userCatAvg_[userDict[u]]))
    feat = np.append(feat, extent2)
    return feat


# In[ ]:

print "start extracting training data"
train_feature = [feature(d['businessID'], d['userID']) for d in train_data]
print "positive feature extracted"
train_feature.extend([feature(b,u) for u,b in negative_pair[:60000]])
train_feature = np.array(train_feature)
train_label = [1 for i in range(140000)]
train_label.extend([0 for i in range(60000)])
train_label = np.array(train_label)
print np.shape(train_feature)
print np.shape(train_label)
test_feature = [feature(b,u) for u,b in test_data]
test_feature = np.array(test_feature)


print "extracting training data complete"
validation_feature = [feature(d['businessID'], d['userID']) for d in validation_data]
validation_feature.extend([feature(b,u) for u,b in negative_pair[60000:100000]])
validation_feature = np.array(validation_feature)
validation_label = [1 for i in range(60000)]
validation_label.extend([0 for i in range(40000)])
validation_label = np.array(validation_label)
print np.shape(validation_feature)
print np.shape(validation_label)

print "extracting feature complete"

# Method using SVM

'''clf = sklearn.svm.SVC(C = 1, kernel = 'rbf')
print "started training"
clf.fit(train_feature, train_label)

print "training complete"

y_predict = clf.predict(validation_feature)
accuracy = np.mean([y_predict == validation_label])

print accuracy
'''
# method using neural network
fc_size = 2
input_size = 5
output_size = 2


def calculate(X, regularizer):
    with tf.variable_scope('fc1'):
        weights1 = tf.get_variable(name='weight', shape=[input_size, fc_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.3))
        bias1 = tf.get_variable(name='bias', shape=[fc_size], initializer=tf.constant_initializer(0.1))
        fc1 = tf.matmul(X, weights1) + bias1
        tf.add_to_collection('losses', regularizer(weights1))

    '''with tf.variable_scope('fc2'):
        weights2 = tf.get_variable(name='weight', shape=[fc_size, output_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.3))
        bias2 = tf.get_variable(name='bias', shape=[output_size], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, weights2) + bias2
        tf.add_to_collection('losses', regularizer(weights2))'''

    return fc1


def train():
    x = tf.placeholder(tf.float32, [None, input_size], name='input_X')
    y = tf.placeholder(tf.int64, [None], name='input_Y')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    y_ = calculate(x, regularizer)
    y_predict = tf.argmax(y_, 1)
    correct_prediction = tf.cast(tf.equal(y_predict, y), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=y))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(max_iter):
            sample = np.random.randint(0, 200000, batch_size)
            x_batch = train_feature[sample]
            y_batch = train_label[sample]
            _, loss_value = sess.run([train_step, loss], feed_dict={x: x_batch, y: y_batch})
            if i % 100 == 0:
                print("After %d iters, loss on training is %f" % (i, loss_value))
                acc = sess.run(accuracy, feed_dict={x: validation_feature, y: validation_label})
                print("After %d iters, accuracy on validation set is %f" % (i, acc))
        print "training complete"
        #write test labels
        predictions = open("predictions_Visit.txt",'w')
        predictions.write("userID-businessID,prediction\n")
        test_label = sess.run(y_predict,feed_dict = {x:test_feature})
        for pair, label in zip(test_data, test_label):
            predictions.write(pair[0] + '-' + pair[1] + ',' + str(label) + '\n')

train()
# method using logistic regression