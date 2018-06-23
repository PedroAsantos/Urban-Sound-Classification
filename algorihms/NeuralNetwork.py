import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree
import sys
import ControlCenter
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

if len(sys.argv)==1:
    features,labels = ControlCenter.getLabelsAndFeatures(1)
else:
    features,labels = ControlCenter.getLabelsAndFeatures(0)

#
# Preparing the training, cross-validation, and test set
#
features,labels = shuffle(features,labels,random_state=0)

sizeOfTrainSet= int(len(features)*0.6)
sizeOfCrossValidationSet = int((len(features)-sizeOfTrainSet)/2)
sizeOfTestSet = len(features)- sizeOfCrossValidationSet - sizeOfTrainSet

train_set_features = features[0:sizeOfTrainSet]
train_set_labels = labels[0:sizeOfTrainSet]

CrossValidation_set_features = features[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]
CrossValidation_set_labels = labels[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]

test_set_features = features[sizeOfTrainSet+sizeOfCrossValidationSet:sizeOfTrainSet+sizeOfCrossValidationSet+sizeOfTestSet]
test_set_labels = labels[sizeOfTrainSet:sizeOfTrainSet+sizeOfCrossValidationSet]


#
# Model methods
#

def MPL_Classifier_1layer(neurons):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(neurons), random_state=1,learning_rate_init=0.01)
	clf.fit(train_set_features, train_set_labels)
	y_pred=clf.predict(CrossValidation_set_features)
	print(y_pred)
	accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
	print("Accuracy MLPClassifier: %f" % (accuracy*100.0))
	return accuracy*100

def MPL_Classifier_2layer(neurons1, neurons2):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(neurons1, neurons2), random_state=1,learning_rate_init=0.01)
	clf.fit(train_set_features, train_set_labels)
	y_pred=clf.predict(CrossValidation_set_features)
	print(y_pred)
	accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
	print("Accuracy MLPClassifier: %f" % (accuracy*100.0))
	return accuracy*100



#
# This is the "main" part of the code
#

neurons = [25, 50, 100, 150, 200, 500]
accuracy_list_1layer = []
accuracy_list_2layer = []

### Testing one hidden layer ###
i = 0
while i < len(neurons):
	print(neurons[i])
	accuracy_list_1layer.append(MPL_Classifier_1layer(neurons[i]))
	i+=1
i = 0
while i < len(accuracy_list_1layer):
	print(accuracy_list_1layer[i])
	i+=1

### Confusion Matrix for 1000 neurons
print("confusion_matrix:")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000), random_state=1,learning_rate_init=0.01)
clf.fit(train_set_features, train_set_labels)
y_pred=clf.predict(CrossValidation_set_features)
print(confusion_matrix(CrossValidation_set_labels, y_pred))

### Ploting results (2D) ###
plt.plot(neurons, accuracy_list_1layer)
plt.ylim(0,100)
plt.scatter(neurons, accuracy_list_1layer)
plt.grid()
plt.xlabel('Number of neurons')
plt.ylabel('Accuracy')
plt.show()


### Testing with two hidden layers ###
i = 0
while i < len(neurons):
	j = 0
	while j < len(neurons):
		print(neurons[i])
		print(neurons[j])
		accuracy_list_2layer.append(MPL_Classifier_2layer(neurons[i], neurons[j]))
		j+=1
	i+=1

i = 0
while i < len(accuracy_list_2layer):
	print(accuracy_list_2layer[i])
	i+=1

### Ploting results (3D) ###

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = [25, 25, 25, 25, 25, 25, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 150, 150, 150, 150, 150, 150, 200, 200, 200, 200, 200, 200, 500, 500, 500, 500, 500, 500]
Y = [25, 50, 100, 150, 200, 500, 25, 50, 100, 150, 200, 500, 25, 50, 100, 150, 200, 500, 25, 50, 100, 150, 200, 500, 25, 50, 100, 150, 200, 500, 25, 50, 100, 150, 200, 500]

ax.scatter(X, Y, accuracy_list_2layer)

ax.set_xlabel('Neurons in first layer')
ax.set_ylabel('Neurons in second layer')
ax.set_zlabel('Accuracy')

plt.show()
