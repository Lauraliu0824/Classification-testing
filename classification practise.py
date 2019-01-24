
# In[1]:


import numpy as np
import sklearn
import time
import random
import time
import os

from sklearn.neighbors import KNeighborsClassifier

#perceptron
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA as sklearnPCA


# In[2]:


def sentimentClassKNN(trainFile, testFile, k, distanceMeasure):
    # Open and read the trainFile and testFile
    start_time = time.time()
    train = ''
    test = ''
    with open(trainFile, 'r') as f:
        train = f.read()

    with open(testFile, 'r') as f:
        test = f.read()

    # Split label and vector
    train_data_list = train.split('\n')
    train_label = [float(data.split()[0]) for data in train_data_list if data != '']
    train_features = [[float(d) for d in data.split()[1:]] for data in train_data_list if data != '']

    test_data_list = test.split('\n')
    test_label = [float(data.split()[0]) for data in test_data_list if data != '']
    test_features = [[float(d) for d in data.split()[1:]] for data in test_data_list if data != '']

    # Create a KNN model
    neigh = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, algorithm='kd_tree', p=2)

    # Fit the training set into the model
    neigh.fit(train_features, train_label) 
    
    # Calculate and return the prediction accuracy
    accuracy = np.sum(neigh.predict(test_features) == np.matrix(test_label)) / float(len(test_label))
    
    print('Accuracy: %f' % accuracy)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return accuracy


# In[3]:


trainFile = 'train_data.txt'
testFile = 'test_data.txt'
k = 10 
distanceMeasure = "euclidean"




# In[4]:


def distance(a, b, distanceMeasure):
    if distanceMeasure == 'euclidean' and len(a) == len(b):
        a = np.matrix(a)
        b = np.matrix(b)
        return np.linalg.norm(a-b)

start_time = time.time()
# Create a KNN model

# Fit the training set into the model

# Calculate and return the prediction accuracy
correct = 0
committee_dists = [float('inf')] * k
committe = [-1] * k

temp = [(0,0)] * len(train_features)
samplesize = 1.0

for j in random.sample(range(1, len(test_features) - 1), samplesize):
    vector = test_features[j]
    for l in range(len(train_features)):
        point = train_features[l]
        dist = distance (vector, point, 'euclidean')
        temp[l] = (train_label[l], dist)
    temp.sort(key=lambda tup: tup[1])
    
    vote = round(sum([pair[0] for pair in temp[:k]])/k, 0)
    if vote == test_label[j]:
        correct += 1
accuracy = correct / samplesize
print("--- %s seconds ---" % (time.time() - start_time))
# In[12]:


knn_accu = sentimentClassKNN('train_data.txt', 'test_data.txt', 10, "euclidean")
assert (0.75<knn_accu)


# In[15]:


def sentimentClassPerceptron(trainFile, testFile, n):
    # Open and read the trainFile and testFile
    # Split label and vector
    # Create a Perceptron model
    # Fit the training set into the model
    # Calculate and return the prediction accuracy
    # open and read file
    start_time = time.time()
    trainFile1 = open(trainFile,'r')
    testFile1 = open(testFile,'r')
    trainString = trainFile1.read()
    testString = testFile1.read()
    
    # tranform the string into arrays
    trainArray = trainString.split('\n')
    testArray = testString.split('\n')
    
    #create empty array to store the values
    newTrainArrayF = []
    newTrainArrayL = []
    newTestArrayF = []
    newTestArrayL = []

    # further split the array and transform all the numbers into float
    for a in trainArray:
        if a != '':
            newTrainArrayL.append(list(map(float,a.split()))[0])
            newTrainArrayF.append(list(map(float,a.split()))[1:])
        
                       
    for b in testArray:
        if b != '':
            newTestArrayL.append(list(map(float,b.split()))[0])
            newTestArrayF.append(list(map(float,b.split()))[1:])
            
    # doing the perceptron 
    ppn = Perceptron(n_iter= 100, eta0=0.001, random_state=0)
    # train data
    ppn.fit(np.array(newTrainArrayF), np.array(newTrainArrayL))
    # predict
    y_pred = ppn.predict(newTestArrayF)
    #calculate accuracy and print it
    print('Accuracy: %f' % accuracy_score(newTestArrayL, y_pred))
    print("Accuracy:",metrics.accuracy_score(newTestArrayL,y_pred))
    print("--- %s seconds ---" % (time.time() - start_time))
    return accuracy_score
    


# In[16]:


per_accu = sentimentClassPerceptron('train_data.txt', 'test_data.txt', 100)
assert (0.8 < per_accu)


# In[14]:


def sentimentClassMyOwn(trainFile, testFile):
    """
    Try three different models
    """
    # Open and read the trainFile and testFile
    # Split label and vector
    # Create three different models
    # Fit the training set into three models
    # open and read file
    start_time = time.time()
    trainFile1 = open(trainFile,'r')
    testFile1 = open(testFile,'r')
    trainString = trainFile1.read()
    testString = testFile1.read()
    
    # tranform the string into arrays
    trainArray = trainString.split('\n')
    testArray = testString.split('\n')
    
    #create empty array to store the values
    newTrainArrayF = []
    newTrainArrayL = []
    newTestArrayF = []
    newTestArrayL = []

    # further split the array and transform all the numbers into float
    for a in trainArray:
        if a != '':
            newTrainArrayL.append(list(map(float,a.split())[0])
            newTrainArrayF.append(list(map(float,a.split())[1:])
        
                       
    for b in testArray:
        if b != '':
            newTestArrayL.append(list(map(float,b.split()))[0])
            newTestArrayF.append(list(map(float,b.split()))[1:])
    
    # create a randomForest Classifier
    clf=RandomForestClassifier(n_estimators=100)
    #train the data
    clf.fit(newTrainArrayF,newTrainArrayL)
    # make prediciton
    y_pred = clf.predict(newTestArrayF)
    #calculate accuracy score
    accuracy2 = metrics.accuracy_score(newTestArrayL,y_pred)
    print("Accuracy:",accuracy2)
    
    time2 = time.time()
    print("--- %s seconds ---" % (time2 - start_time))
    
    # PCA data dimension reduction
    #pca = sklearnPCA(n_components = 25)
    #pca.fit_transform(newTestArrayF)
    #newTrainArrayF2= pca.transform(newTrainArrayF)
    #newTestArrayF2 = pca.transform(newTestArrayF)
    
    # SVC
    gnb = SVC()
    # linear SVC
    #gnb = SVC(kernel="linear", C=0.025)
    gnb.fit(newTrainArrayF,newTrainArrayL)
    pred = gnb.predict(newTestArrayF)
    accuracy3 = metrics.accuracy_score(newTestArrayL,pred)
    print("Accuracy:",accuracy3)
    
    time3 = time.time()
    print("--- %s seconds ---" % (time3 - time2))
    
    
    
    return accuracy1, accuracy2, accuracy3


# In[10]:


accu1, accu2, accu3 = sentimentClassMyOwn('train_data.txt', 'test_data.txt')
assert (0.8 < accu1)
assert (0.8 < accu2)
assert (0.8 < accu3)


# In[11]:


### Check to make sure you include your report.
assert os.path.isfile('Assignment2_Report.pdf')

