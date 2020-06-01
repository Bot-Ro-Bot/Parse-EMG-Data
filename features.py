import librosa
import numpy as np


class Features:
	def __init__(self,words,window_size,sampling_rate,len_max_array):
		self.words = words
		self.window_size = window_size
		self.SR = sampling_rate
		self.step_size = int((window_size/1000) * self.SR)
		self.num_windows = int(len_max_array / self.step_size)
		self.features = []
		self.temp_features = []
		self.spec_features = []
	
	def compute(self):
		self.temp_features = self.compute_temporal_features()
		self.spec_features = self.compute_spectral_features()

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
			self.features.append(f)
			
		print("\n")
		return np.array(self.features)

	def compute_spectral_features(self,word):
		pass

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
