import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.metrics import accuracy_score
import pickle


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
    pickle.dump(objectSerializar,open( "serializableFiles/"+fileName, "wb" ) )

def deserialize(fileName):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    THIS_FOLDER = os.path.join(THIS_FOLDER, "serializableFiles")
    file = os.path.join(THIS_FOLDER, fileName)
    return pickle.load(open(file,"rb"))

def getLabelsAndFeatures(isToSerialize):
    parent_dir = '../Data/UrbanSound8K/audio/'
    #tr_sub_dirs = ["fold1","fold2"]
    sub_dirs=["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8","fold10"]
    #sub_dirs=["fold10"]
    #sub_dirs=["fold5","fold6","fold7","fold8","fold9"]

    #tr_sub_dirs = ["fold1"]
    #ts_sub_dirs = ["fold3"]

    if isToSerialize==1:
        print("WARNNING extracting the features from all the folders that contain the sounds will take hours. To extract only from a few folders or only one change sub_dirs list")
        print("Extracting Features")
    #    tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
    #    ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
    #    serializarList(tr_features,"tr_features")
    #    serializarList(tr_labels,"tr_labels")
    #    serializarList(ts_features,"ts_features")
    #    serializarList(ts_labels,"ts_labels")
        for folder in sub_dirs:
            print("Processing folder_: "+folder)
            features, labels = parse_audio_files(parent_dir,[folder])
            print("SerialigetLabelsAndFeaturesze folder"+ folder)
            serializarList(features,"features_folder_"+folder)
            serializarList(labels,"labels_folder_"+folder)
    else:
        print("Deserialize Features and Labels")
        features=deserialize("features_folder_"+sub_dirs[0])
        labels=np.empty(0)
        featuresTemp=[]
        for folder in sub_dirs:
            print("Deserialize folder_"+folder)
            #features = np.append(deserialize("features_folder_"+folder),features)
            if folder!=sub_dirs[0]:
                featuresTemp = deserialize("features_folder_"+folder)
                features = np.vstack((featuresTemp,features))
            #labelsTemp = deserialize("labels_folder_"+folder)
            labels = np.append(deserialize("labels_folder_"+folder),labels)
    return features,labels
                #features = deserialize("features_folder_"+folder)
                #labels = deserialize("labels_folder_"+folder)
            #ts_features = deserialize("ts_features")
            #ts_labels = deserialize("ts_labels")


        #print(tr_features.size)

        #tr_labels = one_hot_encode(tr_labels)  #why this
        #ts_labels = one_hot_encode(ts_labels)   # why this

##########################################################################################################################
