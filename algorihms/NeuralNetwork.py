import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree
import sys
import ControlCenter
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


if len(sys.argv)==1:
    features,labels = ControlCenter.getLabelsAndFeatures(1)
else:
    features,labels = ControlCenter.getLabelsAndFeatures(0)


features,labels = shuffle(features,labels, random_state=0)

sizeOfTrainSet= int(len(features)*0.6)
sizeOfCrossValidationSet = int((len(features)-sizeOfTrainSet)/2)
sizeOfTestSet = len(features)- sizeOfCrossValidationSet - sizeOfTrainSet


train_set_features = features[0:sizeOfTrainSet]
train_set_labels = labels[0:sizeOfTrainSet]

CrossValidation_set_features = features[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]
CrossValidation_set_labels = labels[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]

test_set_features = features[sizeOfTrainSet+sizeOfCrossValidationSet:sizeOfTrainSet+sizeOfCrossValidationSet+sizeOfTestSet]
test_set_labels = labels[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]



#print(tr_features.shape[1])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(280, 300), random_state=1,learning_rate_init=0.01)
clf.fit(train_set_features, train_set_labels)
y_pred=clf.predict(CrossValidation_set_features)
print(y_pred)
accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
print("Accuracy MLPClassifier: %f" % (accuracy*100.0))
