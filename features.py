import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class Features:
	def __init__(self,words,labels,window_size,sampling_rate,len_max_array):
		self.words = words
		self.labels = labels
		self.window_size = window_size
		self.SR = sampling_rate
		self.step_size = int((window_size/1000) * self.SR)
		self.num_windows = int(len_max_array / self.step_size)
		self.features = []
		self.temp_features = []
		self.spec_features = []
		self.class_names = ["THE","A","TO","OF","IN","ARE","AND","IS","THAT","THEY"]

	
	def compute(self):
		self.temp_features = self.compute_temporal_features()
		self.spec_features = self.compute_spectral_features()
		# for i in range(len(self.words)):
			# self.features.append

	def compute_temporal_features(self):
		print("Extracting temporal features")
		keep = 0
		for word in self.words:
			keep+=1
			if keep%100==0:
				print("-",end=" ")
			f = []
			for i in range(self.num_windows):
				seg = word[i*self.step_size:(i+1)*self.step_size]
				temp = [self.ZCR(seg) , self.ARV(seg) , self.RMS(seg) , self.Mean(seg) ]
				f.extend(temp)
			f.extend(self.MFCC(word).tolist())
			self.temp_features.append(f)
			
		print("\n")
		return np.array(self.temp_features)

	def compute_spectral_features(self):
		print("Extracting Spectral features")
		keep = 0
		for word in self.words:
			keep+=1
			if keep%100==0:
				print("-",end=" ")
			self.spec_features.append(self.MFCC(word))
		print("\n")
		return np.array(self.spec_features)

	def ZCR(self,seg):
		# seg = seg - self.Mean(seg)
		pos = seg>0
		npos = ~pos
		return len(((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0])

	def ARV(self,seg):
		seg= np.abs(seg)
		return np.mean(seg)

	def RMS(self,seg):
		return np.sqrt(np.mean(np.power(seg,2)))

	def Mean(self,seg):
		try:
			return np.mean(seg)
		except:
			return 0

	def MFCC(self,word):
		# random = np.random.randint(0,len(self.words))
		mfcc = librosa.feature.mfcc(y=word,sr=self.SR,n_mfcc=50)
		# print(mfcc)
		mfcc = np.mean(mfcc.T,axis=0)
		return mfcc


	def draw_MFCC(self):
		random = np.random.randint(0,len(self.words))
		x_axis = np.linspace(0,800,len(self.words[0]))

		# plt.figure("Plots of the word "+self.class_names[int(self.labels[random])])
		plt.subplot(311)
		# plt.title("Time domain Representation")
		plt.xlabel("Time (in ms)")
		plt.ylabel("Amplitude")
		plt.plot(x_axis,self.words[random])
		
		plt.subplot(312)
		# plt.title("Frequency domain Representation")
		plt.xlabel("Frequency (in Hz)")
		plt.ylabel("Power")
		plt.magnitude_spectrum(self.words[random],Fs=self.SR)

		plt.subplot(313)
		mfcc = librosa.feature.mfcc(y=self.words[random],sr=self.SR,n_mfcc=40,cmap="RdYlBu")
		librosa.display.specshow(mfcc,sr=self.SR,x_axis="ms",y_axis="mel")
		plt.show()
