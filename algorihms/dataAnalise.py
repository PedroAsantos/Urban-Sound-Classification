import numpy as np
import matplotlib.pyplot as plt
import ControlCenter
import collections
import sys



if len(sys.argv)==1:
    features,labels = ControlCenter.getLabelsAndFeatures(1)
else:
    features,labels = ControlCenter.getLabelsAndFeatures(0)

np.set_printoptions(threshold='nan')

numberOfLabels=[]
print(collections.Counter(labels))
s=collections.Counter(labels)
for i in range(len(s)):
    numberOfLabels.append(s[i])
print(numberOfLabels)
print(sum(numberOfLabels))

plt.bar(range(len(s)), numberOfLabels)
plt.xticks(range(len(s)), range(len(s)))
plt.ylabel('Number of sounds')
plt.xlabel('Classes')
plt.title('Number of sounds files of each class')
plt.show()



sound_file_paths = ["../Data/UrbanSound8K/audio/fold5/100852-0-0-20.wav","../Data/UrbanSound8K/audio/fold10/100648-1-4-0.wav","../Data/UrbanSound8K/audio/fold5/100263-2-0-161.wav"]

sound_names = ["air conditioner","car horn","children playing"]
raw_sounds = ControlCenter.load_sound_files(sound_file_paths)

ControlCenter.plot_waves(sound_names,raw_sounds)
ControlCenter.plot_specgram(sound_names,raw_sounds)
ControlCenter.plot_log_power_specgram(sound_names,raw_sounds)
