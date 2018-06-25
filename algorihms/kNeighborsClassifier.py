import numpy as np
import ControlCenter
from sklearn.naive_bayes import GaussianNB
import sys
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

#######################################
## Use arg -t to test with a lot of inputs in the algorithm
## Use arg -s to extract the features of all the data and serialize the data. If you have already the data serializable run without -s.
## Use arg -p to run preprocessing with StandardScaler
## Use arg -f to run with test set
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
test_set_labels = labels[sizeOfTrainSet+sizeOfCrossValidationSet:sizeOfTrainSet+sizeOfCrossValidationSet+sizeOfTestSet]


#print(sys.argv)

def calculate_confusion_matrix_errors(cm):
    sumErrors = range(0,len(cm))

    for i in range(0,len(cm)-1):
        sumErrors[i] = sum(cm[i,:])
        sumErrors[i]  = sumErrors[i] - cm[i,i]
    return sumErrors


toPreProcList = [x for x in sys.argv if '-p' in x]
toRunTestSet = [x for x in sys.argv if '-f' in x]
if len(toPreProcList)>0 or len(toRunTestSet)>0 :
#    CrossValidation_set_features=ControlCenter.applyPCA(CrossValidation_set_features)
#    test_set_features=ControlCenter.applyPCA(test_set_features)
#    train_set_features=ControlCenter.applyPCA(train_set_features)
     CrossValidation_set_features=ControlCenter.applyStandardScaler(CrossValidation_set_features)
     test_set_features=ControlCenter.applyStandardScaler(test_set_features)
     train_set_features=ControlCenter.applyStandardScaler(train_set_features)

#    CrossValidation_set_features=ControlCenter.applyNormalization(CrossValidation_set_features)
#    test_set_features=ControlCenter.applyNormalization(test_set_features)
#    train_set_features=ControlCenter.applyNormalization(train_set_features)

toTestList = [x for x in sys.argv if '-t' in x]

if len(toTestList)>0:
    saveModel=[]
    accuracys=[]
    accuracysNeigh=[]
    accuracyP=[]
    neighborsList=[1,2,3,4,5,6,7,8,9,10,11]
    neighborsSecList=[1,2,3,4,5]
    pp=[1,2,3]
##testing neighbors
    for neighbors in neighborsList:
        clf = KNeighborsClassifier(n_neighbors=neighbors)
        clf.fit(train_set_features, train_set_labels)

        y_pred=clf.predict(CrossValidation_set_features)
        #print(y_pred)
        accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
        accuracysNeigh.append(accuracy)
        print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))

    plt.plot(neighborsList,accuracysNeigh)
    plt.title('K Neighbors Classifier - Python') # subplot 211 title
    t = plt.xlabel('Number of Neighbors', fontsize=12)
    t = plt.ylabel('Accuracy', fontsize=12)
    plt.show()
    accuracysNeigh=[]
#testing p with neighbor = 1
    for p in pp:
            clf = KNeighborsClassifier(n_neighbors=1,p=p)
            clf.fit(train_set_features, train_set_labels)

            y_pred=clf.predict(CrossValidation_set_features)
            #print(y_pred)
            accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
            accuracyP.append(accuracy)
            print("Nei:"+str(1)+"p:"+str(p)+"acc:"+str(accuracy*100.0))
            print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))
    plt.plot(pp,accuracyP)
    plt.title('K Neighbors Classifier - Python') # subplot 211 title
    t = plt.xlabel('p', fontsize=12)
    t = plt.ylabel('Accuracy', fontsize=12)
    plt.show()
    accuracyP=[]
#testing combinations between neighbors and p
    accuracysNeighPP = [];
    for neighbors in neighborsSecList:
        accuracysTempNeigh=[]
        for p in pp:
            clf = KNeighborsClassifier(n_neighbors=neighbors,p=p)
            clf.fit(train_set_features, train_set_labels)

            y_pred=clf.predict(CrossValidation_set_features)
            #print(y_pred)
            accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
            accuracysTempNeigh.append(accuracy)
            print("nei:"+str(neighbors)+"p:"+str(p)+"acc:"+str(accuracy*100.0))
            print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))
            accuracys.append([accuracy])
            saveModel.append([p])
        accuracysNeighPP.append(accuracysTempNeigh)
    print(accuracysNeighPP)
    maxAccuracyPP=[]
    ppMax=[]
    for listTemp in accuracysNeighPP:
        maxAccuracyPP.append(max(listTemp))
        ppMax.append(listTemp.index(max(listTemp)))
    print(maxAccuracyPP)
    print(ppMax)

if len(toTestList)==0 and len(toRunTestSet)==0:

    clf = KNeighborsClassifier(n_neighbors=1,p=1)
#    clf = KNeighborsClassifier()
    print(clf)
    clf.fit(train_set_features, train_set_labels)

    y_pred=clf.predict(CrossValidation_set_features)
    print(y_pred)
    accuracy = accuracy_score(y_pred, CrossValidation_set_labels)
    #accuracysNeigh.append(accuracy)
    print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))
    c_m=confusion_matrix(CrossValidation_set_labels, y_pred)
    print("confusion_matrix:")
    print(c_m)
    sum_errors = calculate_confusion_matrix_errors(c_m)
    print(sum_errors)
    print(sum(sum_errors))
    plt.bar(range(len(c_m)), sum_errors)
    plt.xticks(range(len(c_m)), range(len(c_m)))
    plt.ylabel('Miss predictions')
    plt.xlabel('Classes')
    plt.title('Number of miss predictions of each class')
    plt.show()



if len(toRunTestSet)>0:
    clf = KNeighborsClassifier(n_neighbors=1,p=1)
#    clf = KNeighborsClassifier()
    print(clf)
    clf.fit(train_set_features, train_set_labels)

    y_pred=clf.predict(test_set_features)
    print(y_pred)
    accuracy = accuracy_score(y_pred, test_set_labels)
    #accuracysNeigh.append(accuracy)
    print("Accuracy KNeighborsClassifier neigh: %f" % (accuracy*100.0))
    c_m=confusion_matrix(test_set_labels, y_pred)
    print("confusion_matrix:")
    print(c_m)
    precision, recall, fscore, support = score(test_set_labels, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

fn="../Data/UrbanSound8K/audio/fold7/101848-9-0-1.wav"
features, labels = np.empty((0,193)), np.empty(0)

mfccs, chroma, mel, contrast,tonnetz = ControlCenter.extract_feature(fn)

ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
features = np.vstack([features,ext_features])

labels = np.append(labels, fn.split('/')[-1].split('-')[1])
feature=np.array(features)
label=np.array(labels, dtype = np.int)

y_pred=clf.predict(feature)
print(y_pred,label)
