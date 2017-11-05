
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

regularization_rate = 0.01
Reduction_Size = 20
batch_size = 200
max_iter = 50000
learning_rate = 0.0001

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


# In[13]:


data = []
for l in readGz('train.json.gz'):
    data.append(l)


# In[14]:


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


# In[15]:


random.shuffle(data)
train_data = data[0:160000]
validation_data = data[160000:200000]
negative_pair = []
randomCnt = 0
while(randomCnt) < 40000:
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


# In[16]:


def Mean(lst):
    if len(lst) != 0:
        return np.mean(lst)
    else:
        return avgRating


# In[17]:


businessRat = [[] for i in range(len(businessList))]
for d in train_data:
    businessRat[businessDict[d['businessID']]].append(d['rating'])
businessAvg = [Mean(businessRat[i]) for i in range(len(businessList))]
print businessAvg[:20]


# In[18]:


categoryList = []
for d in train_data:
    for c in d['categories']:
        if c not in categoryList:
            categoryList.append(c)
categoryDict = defaultdict(int)
for c in categoryList:
    categoryDict[c] = categoryList.index(c)

businessCategory = [[0 for j in range(len(categoryList))] for i in range(len(businessList))]
userHistory = [[0 for j in range(len(categoryList))] for i in range(len(userList))]
for d in train_data:
    for c in d['categories']:
        userHistory[userDict[d['userID']]][categoryDict[c]]+=1
        businessCategory[businessDict[d['businessID']]][categoryDict[c]]=1


# In[32]:


print len(categoryList)


# In[33]:


userFeature = userHistory
upca = sklearn.decomposition.PCA(n_components = 100)
upca.fit_transform(userFeature)


# In[31]:


categoryFeature = [businessCategory[businessDict[d['businessID']]] for d in train_data]
pca = sklearn.decomposition.PCA(n_components = 100)
pca.fit_transform(categoryFeature)


# In[39]:


def feature(b,u):
    feat = [businessAvg[businessDict[b]]]
    feat.extend(userFeature[userDict[u]])
    feat.extend(categoryFeature[businessDict[b]])
    return feat


# In[ ]:


train_feature = [feature(d['businessID'], d['userID']) for d in train_data]
validation_feature = [feature(d['businessID'], d['userID']) for d in validation_data].extend([feature(b,u) for u,b in negative_pair])

