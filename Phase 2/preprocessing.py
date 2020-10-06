import os
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import tqdm
import pprint
import sys
# import warnings
# warnings.filterwarnings("ignore")

import librosa
import librosa.display

# np.set_printoptions(threshold=sys.maxsize)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow import keras

MAIN_DIR = "/home/deadpool/Kaggle Projects/EMG Tanja Ma'am/"
os.chdir(MAIN_DIR)

AUDIBLE = os.path.join("TOP_WORDS","audible")
WHISPERED = os.path.join("TOP_WORDS","whispered")
SILENT = os.path.join("TOP_WORDS","silent")

MODES = [AUDIBLE, WHISPERED, SILENT]

top_10_words = ["THE","A","TO","OF","IN","ARE","AND","IS","THEY","THAT"]
top_20_words = top_10_words + ["THIS","AN","HAS","HE","I","ON","WERE","IT","WE","YOU"]

# for top 10 words
audio_file = "Top10_Audio.pickle"
emg_file = "Top10_EMG.pickle"

# for top 20 words
# audio_file = "Top20_Audio.pickle"
# emg_file = "Top20_EMG.pickle"

audible_audio = pickle.load(open(os.path.join(AUDIBLE,audio_file),"rb"))
audible_emg = pickle.load(open(os.path.join(AUDIBLE,emg_file),"rb"))

whispered_audio = pickle.load(open(os.path.join(WHISPERED,audio_file),"rb"))
whispered_emg = pickle.load(open(os.path.join(WHISPERED,emg_file),"rb"))

silent_audio = pickle.load(open(os.path.join(SILENT,audio_file),"rb"))
silent_emg = pickle.load(open(os.path.join(SILENT,emg_file),"rb"))

LABELS = list(audible_audio.keys())
LABELS.sort()
print(LABELS)

SR_AUDIO = 16000 # in Hz
SR_EMG = 600 # in Hz


FRAME_SIZE = 27 # in milliseconds
FRAME_SHIFT = 10 # in milliseconds

FRAME_SAMPLES = int((FRAME_SIZE/1000)*SR_EMG)
PER_FRAME_SAMPLES = int((FRAME_SHIFT/1000)*SR_EMG)

print(FRAME_SAMPLES)
print(PER_FRAME_SAMPLES)



class EMG(object):
	"""
	preprocessing and feature extraction class for EMG data
	X = List of all instances of input data
	x = an instance of the input data 
	Y = List of all instances of input labels
	y = an instance of the input label
	"""
	"""
	average length of a word utterance: ( 600ms(100 wpm) + 480ms (130 wpm) + 360ms (160 wpm) ) / 3 =  480ms
	"""
	def __init__(self, SR,FRAME_SIZE,FRAME_SHIFT):
		self.SR = SR
		self.FRAME_SIZE = FRAME_SIZE
		self.FRAME_SHIFT = FRAME_SHIFT
	
	def padding(self,X,Y,raw=0,length=12900):
		"""
		length of a phones considered as 30ms(~27ms)
		3 frames per phoneme
		3-5 phonemes per word
		ideal feature length = ( (frame_size * 3 * 5 + frame_size * 3 * 3) /2 ) * SR = 194(~200)
		"""
		
		"""
		length = length of the largest list (length varies with words = 12900)
		"""

		# removing list with null items
		for index,x in enumerate(X):
			if(len(x) == 0 or len(x) < 30):
				del X[index]
				del Y[index]
			elif(len(x)>length):
				X[index] = x[:length]

		for index,x in enumerate(X):
			if(len(x) == length):
				continue
			else:
				diff = length - len(x)
				X[index] = np.pad(X[index], (int(diff/2),int(diff-diff/2)), constant_values=(0,0))
		
		return X,Y

	def raw(self,X,Y):
		if raw:
			length = int( (480/1000)*self.SR)

	def DNPA(self,seg):
		"""Double Nine Point Average"""
		w = []
		for i in range(len(seg)):
			a = i - 4
			b = i + 4
			a = a if a>=0 else 0
			b = b if b<len(seg) else len(seg)
			w.append(int(np.sum(seg[a:b])/9))

		v = []
		for i in range(len(seg)):
			a = i - 4
			b = i + 4
			a = a if a>=0 else 0
			b = b if b<len(seg) else len(seg)
			v.append(int(np.sum(w[a:b])/9))

		return v

	def ZCR(self,seg):
		"""Zero Crossing Rate"""
		pos = seg>0
		npos = ~pos
		return len(((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0])
	
	def HFS(self,seg):
		"""High frequency signals"""
		return np.subtract(seg,self.DNPA(seg))
		
	def RHFS(self,seg):
		"""Rectified High frequency signals"""
		return abs(self.HFS(seg))
	
	def FBP(self,seg):
		"""Frame Based Power"""
		return np.sum(np.power(seg,2))

	def feature(self,seg):
		seg = np.array(seg) * 0.033 #in microvolts
		return np.hstack((self.DNPA(seg),self.RHFS(seg),self.HFS(seg),self.ZCR(seg),self.FBP(seg)))

	def MFCC(self,seg):
		mfcc = librosa.feature.mfcc(y=seg,sr=self.SR,n_mfcc=10)
		return np.mean(mfcc.T,axis=0)

	def STFT(self,seg):
		
		pass

	def features(self,seg):
		seg = np.array(seg) * 0.033 #in microvolts

		pass
	
	def segment(self,x):
		f = []
		for channel in range(6):
			for i in range(len(x[0])):
				a = i*self.FRAME_SHIFT
				b = a + self.FRAME_SIZE
				if(b>len(x[0])):
					break
				seg = x[channel][a:b]
				f.extend(self.feature(seg))
		return f

	def fit(self,X,Y):
		if("X_audible_emg(features).pickle" in os.listdir() ):
			temp_X = pickle.load(open("X_audible_emg(features).pickle","rb"))
		else:
			temp_X = []
			for x,count in zip(X, tqdm.tqdm(range(len(X)),ncols=100) ):
				temp_X.append(self.segment(x))
			
			# save all extracted features to a pickle file
			pickle.dump(temp_X,open("X_audible_emg(features).pickle","wb"))

		return self.padding(temp_X,Y)

	def fit_transform(self,X,Y):
		return self.fit(X,Y)


class AUDIO(object):
	""""preprocessing and feature extraction class for AUDIO data"""
	def __init__(self, SR,FRAME_SIZE,WINDOW_SIZE):
		self.SR = arg

	def padding():
		pass

	def MFCC():
		pass

	def STFT():
		pass


def audio_figures(words =["AND","THAT"]):
	modes = ["Audible Mode","Whisper Mode","Silent Mode"]
	# files_emg = [audible_emg[word],whispered_emg[word],silent_emg[word]] 
	# files_audio = [audible_audio[word],whispered_audio[word],silent_audio[word]]
	# sub = 321
	# plt.figure("THAT_AUDIO")
	# # plt.tight_layout()

	# for mode,file_audio in zip(modes,files_audio):
	# 	abc = file_audio[0]
	# 	# fgh = emg_file[0]
	# 	plt.subplot(sub)
	# 	plt.title(mode)
	# 	plt.xlabel("Samples")
	# 	plt.ylabel("Amplitude")
	# 	plt.ylim([-3500,3500])
	# 	plt.xlim([0,2000])
	# 	plt.plot(abc)
	# 	sub = sub + 1
	
	
	word = words[0]
	files_audio = [audible_audio[word],whispered_audio[word],silent_audio[word]]
	
	fig, axes = plt.subplots(nrows=3,ncols=2)
	for ax,mode,file_audio in zip(axes[:,0],modes,files_audio):
		abc = file_audio[0]
		ax.set_ylabel("Amplitude")
		if mode == "Audible Mode":
			mode = "\""+words[0]+"\""+"\n\n"+ mode
		ax.set_title(mode)
		ax.set_ylim([-3500,3500])
		ax.set_xlim([0,2000])
		ax.plot(abc,"b")
	
	word = words[1]
	files_audio = [audible_audio[word],whispered_audio[word],silent_audio[word]]
	
	for ax,mode,file_audio in zip(axes[:,1],modes,files_audio):
		abc = file_audio[0]
		# ax.set_ylabel("Amplitude")
		if mode == "Audible Mode":
			mode = "\""+words[0]+"\""+"\n\n"+ mode
		ax.set_title(mode)
		ax.set_ylim([-3500,3500])
		ax.set_xlim([0,2000])
		ax.plot(abc,"r")

	axes[0][0].set_xlabel("(a)")
	axes[0][1].set_xlabel("(a)")
	axes[1][0].set_xlabel("(b)")
	axes[1][1].set_xlabel("(b)")
	axes[2][0].set_xlabel("(c)\nSamples")
	axes[2][1].set_xlabel("(c)\nSamples")
	plt.subplots_adjust(top=0.884,bottom=0.11,left=0.073,right=0.986,hspace=0.664,wspace=0.145)
		
	# fig.tight_layout() 

def emg_figures(words =["AND","THAT"]):
	
	modes = ["Audible Mode","Whisper Mode","Silent Mode"]

	for x in range(6):
		word = words[0]
		files_emg = [audible_emg[word],whispered_emg[word],silent_emg[word]] 

		# top=0.892,
		# bottom=0.11,
		# left=0.091,
		# right=0.97,
		# hspace=0.905,
		# wspace=0.445

		fig, axes = plt.subplots(nrows=3,ncols=2)
		# a = "\""+words[0]+"\""+" "*50+"Channel "+str(x+1)+" "*50+"\""+words[1]+"\""
		# fig.suptitle(a)
		fig.suptitle("Channel "+str(x+1))
		fig.legend(["AND","THAT"])
		for ax,mode,file_emg in zip(axes[:,0],modes,files_emg):
			abc = file_emg[20]
			abc = abc[x]
			ax.set_ylabel("Amplitude")
			if mode == "Audible Mode":
				mode = "\""+words[0]+"\""+"\n\n"+ mode
			ax.set_title(mode)
			# ax.set_title(mode+" (\""+word+"\")")
			ax.plot(abc,"b")
		
		word = words[1]
		files_emg = [audible_emg[word],whispered_emg[word],silent_emg[word]] 
		
		for ax,mode,file_emg in zip(axes[:,1],modes,files_emg):
			abc = file_emg[10]
			abc = abc[x]
			# ax.set_ylabel("Amplitude")
			if mode == "Audible Mode":
				mode = "\""+words[1]+"\""+"\n\n"+ mode
			ax.set_title(mode)
			# ax.set_title(mode+" (\""+word+"\")")
			# ax.set_ylim([-3500,3500])
			# ax.set_xlim([0,2000])
			ax.plot(abc,"r")

		# for ax,mode,file_emg in zip(axes[:,0],modes,files_emg):
		# 	abc = file_emg[0]
		# 	abc = abc[x]
		# 	ax.set_ylabel("Amplitude")
		# 	ax.set_title(mode)
		# 	# ax.set_ylim([-3500,3500])
		# 	# ax.set_xlim([0,2000])
		# 	ax.plot(abc,"r",alpha=0.5)

		axes[0][0].set_xlabel("(a)")
		axes[0][1].set_xlabel("(a)")
		axes[1][0].set_xlabel("(b)")
		axes[1][1].set_xlabel("(b)")
		axes[2][0].set_xlabel("(c)\nSamples")
		axes[2][1].set_xlabel("(c)\nSamples")
		# plt.tight_layout() 
		# plt.subplots_adjust(left=0.091,right=0.97,bottom=0.11,top=0.892,hspace=0.905,wspace=0.445)
		plt.subplots_adjust(top=0.884,bottom=0.11,left=0.073,right=0.986,hspace=0.664,wspace=0.145)
		# plt.savefig("Channel "+str(x+1),dpi=500,bbox_inches="tight")
		plt.show()

def check_DNPA():
	feature = EMG(SR_EMG,FRAME_SAMPLES,PER_FRAME_SAMPLES)
	X = audible_emg["THEY"][5]

	X = np.array(X[0])
	x = feature.DNPA(X)
	# plt.title("Double Nine Point Average")
	plt.title("THEY")
	plt.xlabel("Samples")
	plt.ylabel("Amplitude (in uV)")
	plt.plot(X,"b",alpha = 0.3)
	plt.plot(x,"r")
	plt.legend(["Raw Signal","DNPA"])

def check_HFS():
	x = feature.HFS(X)
	print("X ",X[:10])
	print("x ",x[:10])
	print(feature.DNPA(X)[:10])
	plt.title("THEY")
	plt.xlabel("Samples")
	plt.ylabel("Amplitude (in uV)")
	plt.plot(X,"b",alpha=0.5)
	plt.plot(x,"g",alpha=0.8)
	plt.plot(feature.DNPA(X),"r")
	plt.legend(["INPUT SIGNAL","HFS","DNPA"])	

def check_RFHS():
	x = feature.RHFS(X)
	print("X ",X[:10])
	print("x ",x[:10])
	print(feature.HFS(X)[:10])
	plt.title("THEY")
	plt.xlabel("Samples")
	plt.ylabel("Amplitude (in uV)")
	plt.plot(X,"b",alpha=0.3)
	plt.plot(x,"r",alpha=0.8)
	plt.plot(feature.HFS(X),"g",alpha=0.3)
	plt.legend(["INPUT SIGNAL","RHFS","HFS"])	

def reduce_dimension():
	pca_len = 200
	PCA_file = "PCA_X("+ str(pca_len) +").pickle" 
	if(PCA_file in os.listdir()):
		X_new = pickle.load(open(PCA_file,"rb"))

	else:
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		pca = PCA(n_components=pca_len,svd_solver="full")
		X_new = pca.fit_transform(X_scaled)
		print(np.array(X_new).shape)
		pickle.dump(X_new,open(PCA_file,"wb"))

	return X_new

def KNN_Classifier(X_train,y_train,X_test,y_test):
	# print("Training KNN Model")
	KNN_model = KNeighborsClassifier(n_neighbors=8, weights="uniform")
	KNN_model.fit(X_train,y_train)
	KNN_prediction = KNN_model.predict(X_train)
	KNN_accuracy = accuracy_score(y_train,KNN_prediction)
	print("  KNN ACCURACY (Train): ",KNN_accuracy)
	print("  KNN ACCURACY (Test): ",accuracy_score(y_test,KNN_model.predict(X_test)))
	print("")
	plt.matshow(confusion_matrix(y_train,KNN_prediction,normalize="true"),cmap="gray")
	plt.xticks(range(0,10),LABELS)
	plt.yticks(range(0,10),LABELS)

	return KNN_accuracy,accuracy_score(y_test,KNN_model.predict(X_test))

def SVM_Classifier(X_train,y_train,X_test,y_test):
	SVM_model = SVC(verbose=0)
	SVM_model.fit(X_train,y_train)
	SVM_prediction = SVM_model.predict(X_train)
	SVM_accuracy = accuracy_score(y_train,SVM_prediction)
	print("SVM ACCURACY (Train): ",SVM_accuracy)
	print("SVM ACCURACY (Test): ",accuracy_score(y_test,SVM_model.predict(X_test)))
	print("")
	plt.matshow(confusion_matrix(y_train,SVM_prediction,normalize="true"),cmap="gray")
	plt.xticks(range(0,10),LABELS)
	plt.yticks(range(0,10),LABELS)
	return SVM_accuracy, accuracy_score(y_test,SVM_model.predict(X_test))

def K_fold(X_new,Y_new):
	avg_test_acc = []
	avg_train_acc = []
	split = StratifiedShuffleSplit(n_splits=10, test_size =0.1, random_state=42)
	ten_folds = split.split(X_new,Y_new)	
	for i in range(10):
		print("Training CV",i+1)
		train_id, test_id = next(ten_folds) 
		X_train, y_train , X_test, y_test = X_new[train_id], Y_new[train_id], X_new[test_id], Y_new[test_id]
		a,b =KNN_Classifier(X_train,y_train,X_test,y_test)
		# a,b = SVM_Classifier(X_train,y_train,X_test,y_test)
		avg_train_acc.append(a)
		avg_test_acc.append(b)
		# plt.show()
	avg_train_acc = np.array(avg_train_acc)
	avg_test_acc = np.array(avg_test_acc)
	print("Highest Train Accuracy: ", np.amax(avg_train_acc))
	print("Highest Test Accuracy: ", np.amax(avg_test_acc))
	print("Average Train Accuracy: ", np.mean(avg_train_acc))
	print("Average Test Accuracy: ", np.mean(avg_test_acc))


def MLP_Classifier(X_train,y_train,X_test,y_test):

	# y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)
	# print(y)

	MLP_model = keras.Sequential()
	MLP_model.add(keras.layers.Dense(activation="relu",input_shape=X_train[0].shape,units=len(X_train[0])))
	MLP_model.add(keras.layers.Dense(64,activation="relu"))
	MLP_model.add(keras.layers.Dense(32,activation="relu"))
	MLP_model.add(keras.layers.Dense(10,activation="softmax"))

	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	# loss = keras.losses.CategoricalCrossentropy(from_logits=True)
	
	# opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
	opt = keras.optimizers.Adam(lr=0.0001)

	MLP_model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])
	print(MLP_model.summary())
	MLP_model.fit(X_train,y_train,epochs=100)
	
	MLP_prediction = MLP_model.predict_classes(X_train)
	MLP_accuracy = accuracy_score(y_train,MLP_prediction)
	print("MLP ACCURACY (Train): ",MLP_accuracy)
	print("MLP ACCURACY (Test): ",accuracy_score(y_test,MLP_model.predict_classes(X_test)))
	# abc = confusion_matrix(y_train,MLP_prediction,normalize="true")
	# print(np.sum(abc))
	# print(abc)
	plt.matshow(confusion_matrix(y_train,MLP_prediction,normalize="true"),cmap="gray")
	plt.xticks(range(0,10),LABELS)
	plt.yticks(range(0,10),LABELS)
	plt.colorbar()
	
	return MLP_accuracy, accuracy_score(y_test,MLP_model.predict_classes(X_test))


def CNN_Classifier(X_train,y_train,X_test,y_test):
	
	CNN_model = keras.Sequential()
	CNN_model.add(keras.layers.Conv1D(64,kernel_size=3,input_shape=(len(X_train[0]),1),activation="relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
	CNN_model.add(keras.layers.Conv1D(filters=32,kernel_size=3,activation="relu"))
	CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
	CNN_model.add(keras.layers.Flatten())
	CNN_model.add(keras.layers.Dense(100,activation="relu"))
	CNN_model.add(keras.layers.Dense(10,activation="softmax"))
	
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	# opt = keras.optimizers.SGD(lr=0.03, momentum=0.9)
	# opt = keras.optimizers.Adam(lr=0.0001)

	CNN_model.compile(optimizer="adam", loss=loss,metrics=['accuracy'])
	print(CNN_model.summary())
	
	X_train = X_train.reshape(len(X_train),len(X_train[0]),1)
	X_test = X_test.reshape(len(X_test),len(X_test[0]),1)
	CNN_model.fit(X_train,y_train,epochs=25)

	CNN_prediction = CNN_model.predict_classes(X_train)
	CNN_accuracy = accuracy_score(y_train,CNN_prediction)
	print("CNN ACCURACY (Train): ",CNN_accuracy)
	print("CNN ACCURACY (Test): ",accuracy_score(y_test,CNN_model.predict_classes(X_test)))
	plt.matshow(confusion_matrix(y_train,CNN_prediction,normalize="true"),cmap="gray")
	plt.xticks(range(0,10),LABELS)
	plt.yticks(range(0,10),LABELS)

	return CNN_accuracy, accuracy_score(y_test,CNN_model.predict_classes(X_test))


def main():
	X_audible_emg = []
	Y_audible_emg = []

	# X_whispered_emg = []
	# Y_whispered_emg = []
	
	# X_silent_emg = []
	# Y_silent_emg = []
	

	for keys in list(audible_emg.keys()):
		X_audible_emg.extend(audible_emg[keys])
		Y_audible_emg.extend([keys] * len(audible_emg[keys]))
		
		# X_whispered_emg.extend(whispered_emg[keys])
		# Y_whispered_emg.extend([keys] * len(whispered_emg[keys]))
		
		# X_silent_emg.extend(silent_emg[keys])
		# Y_silent_emg.extend([keys] * len(silent_emg[keys]))
		

	print(len(X_audible_emg))
	print(len(Y_audible_emg))

	# print(len(X_whispered_emg))
	# print(len(Y_whispered_emg))

	# print(len(X_silent_emg))
	# print(len(Y_silent_emg))


if __name__ == '__main__':
	pass
	# emg_figures()
	audio_figures()

	# X_audible_emg = []
	# Y_audible_emg = []

	# for keys in list(audible_emg.keys()):
	# 	X_audible_emg.extend(audible_emg[keys])
	# 	Y_audible_emg.extend([keys] * len(audible_emg[keys]))
		
	# print(len(X_audible_emg))
	# print(len(Y_audible_emg))

	# # y_old = pd.Series(Y_audible_emg)
	# # print(y_old.head())
	# # print(y_old.value_counts())

	# f_EMG = EMG(SR_EMG,FRAME_SAMPLES,PER_FRAME_SAMPLES)
	
	# # X, Y = f_EMG.fit_transform(X_audible_emg,Y_audible_emg)
	
	# X, Y = f_EMG.padding(X_audible_emg,Y_audible_emg, raw = True)

	# # y_new = pd.Series(Y)
	# # print(y_new.value_counts())

	# print(len(X))
	# print(len(Y))

	# print(np.array(X).shape)
	# # X_new = reduce_dimension()
	# # X_new = np.array(X)
	# encoder = LabelEncoder()
	# Y_new = encoder.fit_transform(Y)

	# split = StratifiedShuffleSplit(n_splits=1, test_size =0.1, random_state=42)
	# train_id, test_id = next(split.split(X_new,Y_new)) 
	# X_train, y_train , X_test, y_test = X_new[train_id], Y_new[train_id], X_new[test_id], Y_new[test_id]
	
	# print(len(X_train))
	# # MLP_Classifier(X_train,y_train,X_test,y_test)
	# # CNN_Classifier(X_train,y_train,X_test,y_test)
	plt.show()
