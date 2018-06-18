import numpy as np
import ControlCenter
from sklearn.naive_bayes import GaussianNB
import sys
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#######################################
## Use arg -t to test with a lot of inputs in the algorithm
## Use arg -s to extract the features of all the data and serialize the data. If you have already the data serializable run without -s.
########################################



toSerializeList = [x for x in sys.argv if '-s' in x]

if len(toSerializeList)>0:
    features,labels = ControlCenter.getLabelsAndFeatures(1)
else:
    features,labels = ControlCenter.getLabelsAndFeatures(0)

np.set_printoptions(threshold='nan')

features,labels = shuffle(features,labels, random_state=0)

#print(collections.Counter(labels))

sizeOfTrainSet= int(len(features)*0.6)
sizeOfCrossValidationSet = int((len(features)-sizeOfTrainSet)/2)
sizeOfTestSet = len(features)- sizeOfCrossValidationSet - sizeOfTrainSet


train_set_features = features[0:sizeOfTrainSet]
train_set_labels = labels[0:sizeOfTrainSet]

CrossValidation_set_features = features[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]
CrossValidation_set_labels = labels[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]

test_set_features = features[sizeOfTrainSet+sizeOfCrossValidationSet:sizeOfTrainSet+sizeOfCrossValidationSet+sizeOfTestSet]
test_set_labels = labels[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]

neighborsList=[1,2,3,4,5,6,7,8,9,10,11]
accuracysNeigh=[]
print(sys.argv)

toTestList = [x for x in sys.argv if '-t' in x]


if len(toTestList)>0:
    for neighbors in neighborsList:
        clf = KNeighborsClassifier(n_neighbors=neighbors)
        clf.fit(train_set_features, train_set_labels)

        y_pred=clf.predict(CrossValidation_set_features)
        print(y_pred)
        accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
        accuracysNeigh.append(accuracy)
        print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))

    plt.plot(neighborsList,accuracysNeigh)
    plt.title('K Neighbors Classifier - Python') # subplot 211 title
    t = plt.xlabel('Number of Neighbors', fontsize=12)
    t = plt.ylabel('Accuracy', fontsize=12)

    plt.show()
else:

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(train_set_features, train_set_labels)

    y_pred=clf.predict(CrossValidation_set_features)
    print(y_pred)
    accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
    accuracysNeigh.append(accuracy)
    print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))






fn="../Data/UrbanSound8K/audio/fold10/102857-5-0-18.wav"
features, labels = np.empty((0,193)), np.empty(0)

mfccs, chroma, mel, contrast,tonnetz = ControlCenter.extract_feature(fn)

ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
features = np.vstack([features,ext_features])

labels = np.append(labels, fn.split('/')[-1].split('-')[1])
feature=np.array(features)
label=np.array(labels, dtype = np.int)

y_pred=clf.predict(feature)
print(y_pred,label)
