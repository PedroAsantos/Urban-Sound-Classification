import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import sys
import pickle
from sklearn import tree
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=18)
    plt.show()


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print "Error encountered while parsing file: ", fn
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            print(fn)
            labels = np.append(labels, fn.split('/')[-1].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def serializarList(objectSerializar,fileName):
    pickle.dump(objectSerializar,open( fileName, "wb" ) )

def deserialize(fileName):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(THIS_FOLDER, fileName)
    return pickle.load(open(file,"rb"))

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')
parent_dir = '../Data/UrbanSound8K/audio/'
#tr_sub_dirs = ["fold1","fold2"]
tr_sub_dirs = ["fold1"]
ts_sub_dirs = ["fold3"]

if len(sys.argv)==1:
    print("serialize")
    tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
    ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
    serializarList(tr_features,"tr_features")
    serializarList(tr_labels,"tr_labels")
    serializarList(ts_features,"ts_features")
    serializarList(ts_labels,"ts_labels")
else:
    print("deserialize")
    tr_features = deserialize("tr_features")
    tr_labels = deserialize("tr_labels")
    ts_features = deserialize("ts_features")
    ts_labels = deserialize("ts_labels")


#print(tr_features.size)
#tr_labels = one_hot_encode(tr_labels)  #why this
#ts_labels = one_hot_encode(ts_labels)   # why this

##########################################################################################################################


clf = tree.DecisionTreeClassifier()
clf.fit(tr_features.tolist(), tr_labels.tolist())

y_pred=clf.predict(ts_features.tolist())
print(y_pred)
accuracy = accuracy_score(y_pred, ts_labels)
print("Accuracy DecisionTreeClassifier: %f" % (accuracy*100.0))



#classif = OneVsRestClassifier(estimator=SVC(random_state=0))
#y_pred=classif.fit(tr_features.tolist(), tr_labels.tolist()).predict(ts_features.tolist())
#print(y_pred)
#accuracy = accuracy_score(y_pred, ts_labels)
#print("Accuracy OneVsRestClassifier: %f" % (accuracy*100.0))


print(tr_features.shape[1])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(280, 300), random_state=1,learning_rate_init=0.01)
y_pred=clf.fit(tr_features.tolist(), tr_labels.tolist()).predict(ts_features.tolist())
print(y_pred)
accuracy = accuracy_score(y_pred, ts_labels)
print("Accuracy MLPClassifier: %f" % (accuracy*100.0))
