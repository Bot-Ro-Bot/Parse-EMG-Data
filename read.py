from scipy.io import wavfile
import os 
import glob
import numpy as np 
import matplotlib.pyplot as plt 

main_dir = os.getcwd()
filename = "*/*.wav" 

files = []

"""
reading all the audio filenames into a list
"""
for each in glob.glob(filename):
	files.append(each)

"""
reading the audio files and their labels into a list of arrays 
"""
labels = []
words = []
length = []
for file in files:
	labels.append(file.split("/")[0])
	SR, audio = wavfile.read(file)
	words.append(audio)
	length.append(len(audio))

# print(labels[-1])
# print(type(words[-1]))
# print(SR)
print(max(length))
# print(min(length))
len_max_array = max(length)
len_min_array = min(length)


"""
zero padding all the audio files so that each file equal number of members in the array
"""
for i in range(len(words)):
	diff = len_max_array - len(words[i])
	if(diff>0):
		words[i] = np.pad( words[i] ,(int(diff/2),int(diff-diff/2)))



class_names = ["THE","A","TO","OF","IN","ARE","AND","IS","THAT","THEY"]
# print(labels)


# map_labels = lambda label: label for 


print(labels[1:100])
print(type(labels[0]))
for i,label in enumerate(labels):
	
	if label == "THE":
		labels[i]= 0

	if label == "A":
		labels[i]= 1

	if label == "TO":
		labels[i]= 2
	
	if label == "OF":
		labels[i]= 3
	
	if label == "IN":
		labels[i]= 4
	
	if label == "ARE":
		labels[i]= 5
	
	if label == "AND":
		labels[i]= 6
	
	if label == "IS":
		labels[i]= 7
	
	if label == "THAT":
		labels[i]= 8
	
	if label == "THEY":
		labels[i]= 9

	

print(labels[1:100])
print(type(labels[0]))