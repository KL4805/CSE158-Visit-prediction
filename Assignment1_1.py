
# coding: utf-8

# In[3]:


import numpy as np
import scipy as sp
import sklearn
import nltk
import tensorflow as tf
import matplotlib
import gzip
import math
import random
from collections import defaultdict



def readGz(f):
  for l in gzip.open(f):
    yield eval(l)
#trying a classification method using something like Jaccard similarity


# In[4]:


data = []
for l in readGz('train.json.gz'):
    data.append(l)


# In[5]:


random.shuffle(data)
train_data = data[:160000]
validation_data = data[160000:200000]
trainUserList = []
userList = []
trainBusinessList = []
businessList = []


# In[6]:


visitedDict = defaultdict(int)
for d in train_data:
    if d['businessID'] not in trainBusinessList:
        trainBusinessList.append(d['businessID'])
        businessList.append(d['businessID'])
    if d['userID'] not in trainUserList:
        trainUserList.append(d['userID'])
        userList.append(d['userID'])
    visitedDict[(d['userID'], d['businessID'])] = 1
for d in validation_data:
    if d['businessID'] not in businessList:
        businessList.append(d['businessID'])
    if d['userID'] not in userList:
        userList.append(d['userID'])


# In[7]:


userDict = defaultdict(int)
businessDict = defaultdict(int)
for i in range(len(userList)):
    userDict[userList[i]] = i
for i in range(len(businessList)):
    businessDict[businessList[i]] = i


# In[8]:


negative_pair = []
cnt = 0

while cnt < 120000:
    u = random.randint(0, len(trainUserList)-1)
    b = random.randint(0, len(trainBusinessList)-1)
    if visitedDict[(userList[u], businessList[b])] == 0:
        negative_pair.append((u,b))
        cnt+=1
#Sampling negative pairs, note that only sample from training_data


# In[9]:


#building lists of user's visited businesses
u_visited = [defaultdict(float) for u in trainUserList]
b_visited = [defaultdict(float) for b in trainBusinessList]
for d in train_data:
    u = userDict[d['userID']]
    b = businessDict[d['businessID']]
    rating = d['rating']
    u_visited[u][b] = rating
    b_visited[b][u] = rating


# In[10]:


#Get average rating of all businesses and all users
userAvg = [np.mean([u_visited[u][t] for t in u_visited[u]]) for u in range(len(trainUserList))]
businessAvg = [np.mean([b_visited[b][t] for t in b_visited[b]]) for b in range(len(trainBusinessList))]
avgRating = np.mean([d['rating'] for d in train_data])


# In[11]:


def Jaccard(b1, b2):
    b1Set = set([u for u in b_visited[b1]])
    #print b1Set
    b2Set = set([u for u in b_visited[b2]])
    #print b2Set
    return (len(b1Set & b2Set)*1.0)/len(b1Set | b2Set)


# In[12]:


def uJaccard(u1, u2):
    u1Set = set([b for b in u_visited[u1]])
    u2Set = set([b for b in u_visited[u2]])
    return (len(u1Set & u2Set)*1.0)/len(u1Set | u2Set)


# In[13]:


def Pearson(b1,b2):
    b1Set = set([u for u in b_visited[b1]])
    b2Set = set([u for u in b_visited[b2]])
    b1rList = []
    b2rList = []
    uavg = []
    for u in (b1Set & b2Set):
        b1rList.append(b_visited[b1][u])
        b2rList.append(b_visited[b2][u])
        uavg.append(userAvg[u])

    if len(b1Set & b2Set) != 0:
        cov = np.sum([(b1rList[i]-uavg[i])*(b2rList[i]-uavg[i]) for i in range(len(b1rList))])
        std = math.sqrt(np.sum([(r-a)**2 for r,a in zip(b1rList, uavg)]) * np.sum(([(r-a)**2 for r,a in zip(b2rList, uavg)])))
        return (cov*1.0)/std if std != 0 else 0
    else:
        return 0


# In[14]:


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


# In[42]:


'''uMostJaccard = [[]for u in trainUserList]
uMostPearson = [[] for u in trainUserList]
uLeastPearson = [[] for u in trainUserList]
for u in range(len(trainUserList)):
    for v in range(len(trainUserList)):
        uSet = set([b for b in u_visited[u]])
        vSet = set([b for b in u_visited[v]])
        if len(uSet & vSet)>0:
            jac = uJaccard(u,v)
            if jac != 0:
                uMostJaccard[u].append((jac, v))
    uMostJaccard[u].sort()
    uMostJaccard[u].reverse()
    uMostJaccard[u] = uMostJaccard[u][:3]

print uMostJaccard[:10]'''


# In[15]:


#Get general popularity of business and user
businessVisited = [0 for b in trainBusinessList]
userActivity = [0 for u in trainUserList]
for d in train_data:
    u = userDict[d['userID']]
    b = businessDict[d['businessID']]
    businessVisited[b] += 1
    userActivity[u] += 1
userActivity = np.array(userActivity)/(np.max(userActivity)*1.0)
businessVisited = np.array(businessVisited)/(np.max(businessVisited))


# In[16]:


#use dictionary to calculate each user's visit popularity
userVisitTimes = [defaultdict(int) for u in trainUserList]
for d in train_data:
    u = userDict[d['userID']]
    b = businessDict[d['businessID']]
    userVisitTimes[u][b] += 1
mostFrequent = [[] for u in trainUserList]
mostFrequent = [[(u[b],b) for b in u] for u in userVisitTimes]
for u in mostFrequent:
    u.sort()
    u.reverse()


# In[17]:


#Get users most rated business and least rated business
userMostRated = [[(u[b],b) for b in u] for u in u_visited]

for u in userMostRated:
    u.sort()
    u.reverse()


# In[18]:


#add a feature: the top 3 Jaccard Similarity
def feature(u,b):
    if u < len(trainUserList) and b < len(trainBusinessList):
        visitedBusiness = u_visited[u]
        feat = [businessAvg[b]-avgRating, userActivity[u], businessVisited[b]]
        feat.append(Jaccard(b, mostFrequent[u][0][1]))
        feat.append(Jaccard(b, mostFrequent[u][-1][1]))
        feat.append(Jaccard(b, userMostRated[u][0][1]))
        feat.append(Jaccard(b, userMostRated[u][-1][1]))
        JaccardList = []
        for b_ in u_visited[u]:
            JaccardList.append(Jaccard(b,b_))
        JaccardList.sort()
        JaccardList.reverse()
        feat.append(np.mean(JaccardList))
        if(len(JaccardList)>0):
            feat.append(JaccardList[0])
        else:
            feat.append(0)
        if(len(JaccardList)>1):
            feat.append(JaccardList[1])
        else:
            feat.append(0)
        if(len(JaccardList)>2):
            feat.append(JaccardList[2])
        else:
            feat.append(0)
        if(len(JaccardList)>3):
            feat.append(JaccardList[3])
        else:
            feat.append(0)
        feat.append(Pearson(b, mostFrequent[u][0][1]))
        feat.append(Pearson(b, mostFrequent[u][-1][1]))
        feat.append(Pearson(b, userMostRated[u][0][1]))
        feat.append(Pearson(b, userMostRated[u][-1][1]))
        PearsonList = []
        for b_ in u_visited[u]:
            PearsonList.append(Pearson(b,b_))
        feat.append(np.mean(PearsonList))
        PearsonList.sort()
        PearsonList.reverse()
        if(len(PearsonList)>0):
            feat.append(PearsonList[0])
        else:
            feat.append(0)
        if(len(PearsonList)>1):
            feat.append(PearsonList[1])
        else:
            feat.append(0)
        if(len(PearsonList)>2):
            feat.append(PearsonList[2])
        else:
            feat.append(0)
        if(len(PearsonList)>3):
            feat.append(PearsonList[3])
        else:
            feat.append(0)
        if(len(PearsonList)>0):
            feat.append(PearsonList[-1])
        else:
            feat.append(0)
        if(len(PearsonList)>1):
            feat.append(PearsonList[-2])
        else:
            feat.append(0)
        #feat.append(np.dot(gammau_[u], gammai_[b]))
        return feat
    elif u < len(trainUserList):
        feat = [0 for i in range(23)]
        feat[1] = userActivity[u]
        return feat
    elif b < len(trainBusinessList):
        feat = [0 for i in range(23)]
        feat[0] = businessAvg[b]
        feat[2] = businessVisited[b]
        return feat
    else:
        return [0 for i in range(23)]


# In[19]:


train_feature = [feature(userDict[d['userID']], businessDict[d['businessID']]) for d in train_data]
train_feature.extend([feature(u,b) for u,b in negative_pair[:80000]])
validation_feature = [feature(userDict[d['userID']], businessDict[d['businessID']]) for d in validation_data]
validation_feature.extend([feature(u,b) for u,b in negative_pair[80000:120000]])
train_label = [1 for i in range(160000)]
train_label.extend([0 for i in range(80000)])
validation_label = [1 for i in range(40000)]
validation_label.extend([0 for i in range(40000)])
train_feature = np.array(train_feature)
train_label = np.array(train_label)
validation_feature = np.array(validation_feature)
validation_label = np.array(validation_label)
print "Feature extracted"


# In[20]:


test_data = []
predictions = open("predictions_Visit.txt", 'w')
for l in open("pairs_Visit.txt"):
    if l.startswith('userID'):
        continue
    else:
        u,b = l.strip().split('-')
        test_data.append((u,b))


# In[21]:


test_feature = np.array([feature(userDict[u],businessDict[b]) for u,b in test_data])


# In[38]:


batch_size = 300
regularization_rate = 0.1
input_size = 23
output_size = 2
learning_rate = 0.00001

max_iter = 30000



# In[23]:



def calc(X, regularizer):
    with tf.variable_scope("log"):
        w = tf.get_variable(name = 'weight', shape = [input_size, output_size], initializer = tf.truncated_normal_initializer(stddev = 0.05))
        b = tf.get_variable(name = 'bias', shape = [output_size], initializer = tf.constant_initializer(0.05))
        log = tf.matmul(X,w)+b
        tf.add_to_collection('losses',regularizer(w))
    return log


# In[24]:



X = tf.placeholder(tf.float32, [None, input_size], name = 'x-input')
y = tf.placeholder(tf.int64, [None], name = 'y-input')
regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
y_ = calc(X, regularizer)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_, labels = y)
loss = tf.reduce_mean(cross_entropy)+tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
y_predict = tf.argmax(y_,1)
correct = tf.cast(tf.equal(y_predict,y), tf.float32)
accuracy = tf.reduce_mean(correct)


# In[40]:


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(max_iter):
        sample = np.random.randint(0,240000,batch_size)
        x_batch = train_feature[sample]
        y_batch = train_label[sample]
        _, loss_value = sess.run([train_op, loss], feed_dict = {X:x_batch, y:y_batch})
        if i % 200 == 0:
            print("After %d iters, loss on training batch is %f"%(i, loss_value))  
            acc = sess.run(accuracy, feed_dict = {X:validation_feature, y:validation_label})
            print("After %d iters, accuracy on validation is %f"%(i, acc))
    predictions = open("predictions_Visit.txt",'w')
    predictions.write("userID-businessID,prediction\n")
    test_label = sess.run(y_predict,feed_dict = {X:test_feature})
    for pair, label in zip(test_data, test_label):
        predictions.write(pair[0] + '-' + pair[1] + ',' + str(label) + '\n')

