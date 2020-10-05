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
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
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
window_size = 30 #in milliseconds
# seg_size = 160 #equivalent sample size of the window_size

words,labels,SR,len_max_array = read_data()
"""
mapping words in labels to numbers
"""
labels = np.array(labels)
le = LabelEncoder()
# le.fit(labels)
labels= to_categorical(le.fit_transform(labels))
# labels = tf.keras.utils.to_categorical(labels,num_classes=10)
# print(labels)
# print(le.classes_)
# print(le.transform(labels))
# labels.astype(float)
# print("Labels ko type",type(labels))
# print("Labels ko type",type(labels[0]))
# print(labels[0])


f = Features(words,labels,window_size,SR,len_max_array)
# f.draw_MFCC(s)

feat = (f.compute_temporal_features())
# feat = (f.compute_spectral_features())
# print(feat)
print(feat.shape)

"""
shuffling the data
"""

# print(labels[0:100])

def shuffle(feat,labels):
	limit = len(labels)
	random_index = np.random.permutation(limit)
	# print(random_index)
	return feat[random_index],labels[random_index]

feat, labels = shuffle(feat,labels)
feat, labels = shuffle(feat,labels)
# print(len(feat[0]))


train_data, test_data, train_label, test_label = train_test_split(feat,labels,test_size=0.20)
print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

print(train_label[100])
print(test_label[50])


# pred = model.predict(test_data)
# print((prediction[50]))
# print("prediction should be: " , test_label[50])
# points = 0 

# for i in range(len(pred)):
# 	# print(int(np.argmax(pred[i])),int(test_label[i]))
# 	if ( int(np.argmax(pred[i])) == int(test_label[i]) ):
# 		points+=1

# print("Accuracy : ", ( (points/ len(pred)) * 100))


def train_temp():
	model = keras.Sequential()
	model.add(keras.layers.Dense(len(feat[0]) , activation = "relu" , input_shape=(len(feat[0]),)))
	model.add(keras.layers.Dropout(0.25))
	# model.add(keras.layers.Dense(128 , activation = "relu"))
	# model.add(keras.layers.Dropout(0.05))
	model.add(keras.layers.Dense(64 , activation = "relu"))
	# model.add(keras.layers.Dropout(0.10))
	model.add(keras.layers.Dense(32 , activation = "relu"))
	# model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(10 , activation = "softmax"))

	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	history = model.fit(train_data,train_label,validation_split = 0.20 , epochs=150,batch_size=200)

	model.summary()

	test_loss, test_accuracy = model.evaluate(test_data,test_label)
	print("Tested accuracy: ", test_accuracy)
	print("Tested loss: ", test_loss)
	# plt.plot(history.history["accuracy"])
	# plt.plot(history.history["val_accuracy"])
	# plt.plot(history.history["loss"])
	plt.show()

def train_spec():
	model = keras.Sequential()
	model.add(keras.layers.Dense(len(feat[0]) , activation = "relu" , input_shape=(len(feat[0]),)))
	# model.add(keras.layers.Dropout(0.30))
	model.add(keras.layers.Dense(64 , activation = "relu"))
	# model.add(keras.layers.Dropout(0.10))
	# model.add(keras.layers.Dense(64 , activation = "relu"))
	# model.add(keras.layers.Dropout(0.10))
	model.add(keras.layers.Dense(32 , activation = "relu"))
	# model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(10 , activation = "softmax"))

	model.compile(optimizer='adam',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(train_data,train_label,epochs=300,batch_size=400)

	model.summary()

	test_loss, test_accuracy = model.evaluate(test_data,test_label)
	print("Tested accuracy: ", test_accuracy)
	print("Tested loss: ", test_loss)
 
# train_spec()
train_temp()
