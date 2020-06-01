import pandas as pd 
import sklearn
import numpy as np 
from features import *
import os
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt 
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras


def read_data(zero_padding=True,normalize=True):
	main_dir = os.getcwd()
	filename = "*/*.wav" 
	files = []
	labels = []
	words = []
	length = []
	SR = 0
	"""
	reading all the audio filenames into a list
	"""
	for file in glob.glob(filename):
		files.append(file)
		labels.append(file.split("/")[0])
		SR, audio = wavfile.read(file)
		words.append(audio)
		length.append(len(audio))

	len_max_array = max(length)
	len_min_array = min(length)
	
	"""
	zero padding all the audio files so that each file equal number of members in the array
	"""
	if(zero_padding):
		for i in range(len(words)):
			diff = len_max_array - len(words[i])
			if(diff>0):
				words[i] = np.pad( words[i] ,(int(diff/2),int(diff-diff/2)))
	if(normalize):
		words = [zscore(word) for word in words]

	return words,(labels),SR,len_max_array

class_names = ["THE","A","TO","OF","IN","ARE","AND","IS","THAT","THEY"]
window_size = 10 #in milliseconds
# seg_size = 160 #equivalent sample size of the window_size

words,labels,SR,len_max_array = read_data()
"""
mapping words in labels to numbers
"""

for i,label in enumerate(labels):
	
	if label == "THE":
		labels[i]= 0.0

	if label == "A":
		labels[i]= 1.0

	if label == "TO":
		labels[i]= 2.0
	
	if label == "OF":
		labels[i]= 3.0
	
	if label == "IN":
		labels[i]= 4.0
	
	if label == "ARE":
		labels[i]= 5.0
	
	if label == "AND":
		labels[i]= 6.0
	
	if label == "IS":
		labels[i]= 7.0
	
	if label == "THAT":
		labels[i]= 8.0
	
	if label == "THEY":
		labels[i]= 9.0



labels = np.array(labels)
# labels.astype(float)
# print("Labels ko type",type(labels))
# print("Labels ko type",type(labels[0]))
# print(labels[0])


f = Features(words,window_size,SR,len_max_array)
feat = (f.compute_temporal_features())


train_data, test_data, train_label, test_label = train_test_split(feat,labels,test_size=0.20)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

print(train_label[100])
print(test_label[50])

model = keras.Sequential()
model.add(keras.layers.Dense(340 , activation = "relu" , input_shape=(340,)))
# model.add(keras.layers.Dropout(0.01))
model.add(keras.layers.Dense(240 , activation = "relu"))
# model.add(keras.layers.Dropout(0.10))
model.add(keras.layers.Dense(120 , activation = "relu"))
# model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Dense(64 , activation = "relu"))
model.add(keras.layers.Dense(10 , activation = "softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,train_label,epochs=50,batch_size=500)

test_loss, test_accuracy = model.evaluate(test_data,test_label)
print("Tested accuracy: ", test_accuracy)
print("Tested loss: ", test_loss)

pred = model.predict(test_data)
# print((prediction[50]))
# print("prediction should be: " , test_label[50])
points = 0 

for i in range(len(pred)):
	print(int(np.argmax(pred[i])),int(test_label[i]))
	if ( int(np.argmax(pred[i])) == int(test_label[i]) ):
		points+=1

print("Accuracy : ", ( (points/ len(pred)) * 100))