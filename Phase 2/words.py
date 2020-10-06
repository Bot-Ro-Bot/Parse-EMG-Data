import os
import glob
import sys
from scipy.io import wavfile
import struct
import pickle
import numpy as np

# defined directories
MAIN_DIR = "/home/deadpool/Kaggle Projects/EMG Tanja Ma'am/"
AUDIO = "audio"
EMG = "emg"
TRANS = "Transcripts"
ALIGN = "Alignments"
OFFSET = "offset"
CATEGORY = "Subsets"

# directories to be defined
DATASET = "Dataset"
SILENT = "silent"
WHISPERED = "whispered"
AUDIBLE = "audible"

MODES =[AUDIBLE,WHISPERED,SILENT] 

TOP_WORDS = os.path.join(MAIN_DIR,"TOP_WORDS") 
# TOP_10 = os.path.join(MAIN_DIR,"TOP_10")
# TOP_20 = os.path.join(MAIN_DIR,"TOP_10")

os.makedirs(TOP_WORDS,exist_ok=True)
# os.makedirs(TOP_20,exist_ok=True)
os.makedirs(os.path.join(TOP_WORDS,MODES[0]),exist_ok=True)
os.makedirs(os.path.join(TOP_WORDS,MODES[1]),exist_ok=True)
os.makedirs(os.path.join(TOP_WORDS,MODES[2]),exist_ok=True)

top_10_words = ["THE","A","TO","OF","IN","ARE","AND","IS","THEY","THAT"]
top_20_words = top_10_words + ["THIS","AN","HAS","HE","I","ON","WERE","IT","WE","YOU"]



def get_data(glob_audio,glob_emg):
	all_audible = []
	all_emg = []
		
	for audio, emg in zip(glob_audio,glob_emg):
		SR, file = wavfile.read(audio)
		
		with open(emg,"rb") as raw_emg:
			signal = raw_emg.read()
			h_endian = 'h'*int(len(signal) / 2)
			values = np.array(struct.unpack('<'+h_endian,signal))
			values = values.reshape(int(len(signal) / (2*7)),7)
			values = values.T

		temp_audio, temp_emg = apply_offset(os.path.basename(audio),file[:,0],values)
		all_audible.append(temp_audio)
		all_emg.append(temp_emg)
	
	return SR,all_audible,all_emg


def apply_offset(name,file_audio,file_emg):
	new_file_emg = []
	name = "offset_" +name.replace("-","_")[:-4]+ ".txt"
	name = (glob.glob(OFFSET+"/*/*/"+name))[0]

	with open(name,"r") as text:
		offset = (text.read()).split("\n")
		# print(offset)
		file_audio = file_audio[int(offset[0].split(" ")[0]):int(offset[0].split(" ")[1])]
		for channel in range(6):
			new_file_emg.append(file_emg[channel,int(offset[1].split(" ")[0]):int(offset[1].split(" ")[1])])
	
	return file_audio,np.array(new_file_emg)


def pick_words(all_audible, all_emg, glob_audio, glob_emg,top_words=top_10_words):

	emg_dict = {}
	audio_dict= {}
	emg_list =[]
	audio_list = []

	for w in top_words:
		emg_dict[w]= []
		audio_dict[w] = []

	for audio,emg,file_audio,file_emg in zip(glob_audio,glob_emg,all_audible,all_emg):
		name = os.path.basename(audio)
		name = "words_" +name.replace("-","_")[:-4]+ ".txt"	
		name = (glob.glob(ALIGN+"/*/*/"+name))[0]
		
		with open(name,"r") as align:
			words = align.read().split("\n")[:-1]

			for word in words:
				folder = word.split(" ")[-1]

				if(folder in top_words):
					A = int(word.split(" ")[0])
					B = int(word.split(" ")[1])
					
					# pad 2frames at each end of the word (2*160=320 samples)
					a = A * 160 - 320
					a = a if a >0 else 0
					b = B * 160 + 320
					# print((b-a))
					# audio_list.append(file_audio[a:b])
					audio_dict[folder].append(file_audio[a:b])
					# print(folder)

					c = A * 6 - 12
					c = c if c>=0 else 0
					d = B * 6 + 12

					for channel in range(6):
						# 7th channel is just a marker signal
						emg_list.append(file_emg[channel,c:d])

					emg_dict[folder].append(emg_list)
					emg_list = []
		# break
	return audio_dict,emg_dict


def main(top_words=top_10_words):

	for mode in MODES:
		glob_audio = glob.glob(os.path.join(MAIN_DIR,DATASET,mode,AUDIO)+"/*.wav")
		glob_emg = glob.glob(os.path.join(MAIN_DIR,DATASET,mode,EMG)+"/*.adc")
			
		glob_audio.sort()
		glob_emg.sort()

		SR, all_audible,all_emg = get_data(glob_audio, glob_emg)
		audio_dict,emg_dict = pick_words(all_audible, all_emg, glob_audio, glob_emg,top_words)

		# pickle.dump(all_audible,open(os.path.join(TOP_10,mode,"ASR_Audio.picke"),"wb"))
		# pickle.dump(all_emg,open(os.path.join(TOP_10,mode,"ASR_EMG.picke"),"wb"))
		pickle.dump(audio_dict,open(os.path.join(TOP_WORDS,mode,"Top"+ str(len(top_words)) +"_Audio.pickle"),"wb"))
		pickle.dump(emg_dict,open(os.path.join(TOP_WORDS,mode,"Top"+ str(len(top_words)) +"_EMG.pickle"),"wb"))



if __name__ == '__main__':
	os.chdir(MAIN_DIR)
	main(top_10_words)
	main(top_20_words)
	# pass
	
