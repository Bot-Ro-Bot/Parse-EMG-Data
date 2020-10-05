import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import glob
# import xlwt

audio_folder = "audio"
transcript_folder = "Transcripts"
destination_folder = "Extracted Words"
main_dir = os.getcwd()


# print()

# if (not(destination_folder in os.listdir())):
# 	os.makedirs(destination_folder)
# else:
# 	print("The folder already exists")


# os.chdir(transcript_folder)
# print(os.getcwd())
# file = open("transcript_002_001_0100.txt","r")
# transcript = file.read()
# transcript = np.array(transcript[0:-1].split(" "))
# print(transcript)


class Parser:
	def __init__(self):
		pass


def get_files(path=transcript_folder,main =main_dir ):
	os.chdir(path)
	files = []
	for file in glob.glob("*/*/transcript*.txt"):
		files.append(file)
	return files

def read_file(file):
	f = open(file,"r")
	transcript = f.read()
	transcript = transcript[0:-1].split(" ")
	# print(transcript)
	return transcript

def get_all(path=transcript_folder,main =main_dir):
	os.chdir(path)
	files = []
	words= []
	for file in glob.glob("*/*/transcript*.txt"):
		words.extend(read_file(file))
	os.chdir(main_dir)
	return words	

def count_words(words):
	words.sort()
	freq = []
	i = 0
	loop_count = (len(words))
	new_words = []

	while (i < loop_count):
		f = (words.count(words[i]))
		new_words.append(words[i])
		i+= int(f)
		freq.append(f)

	return new_words,freq

def verify(words):
	#dictionary ko number of words sanga match bhairako xaina, tesaile test matra gareko
	print(("ABUSE" in words)) #false return garxa, so dataset maa xaina tyo word
	print(("STAFF" in words)) #true return garxa, so dataset maa xa tyo word
	print(("HEAD" in words)) #true return garxa, so dataset maa xa tyo word
	print(("YOUNGSTERS" in words))#false return garxa, so dataset maa xaina tyo word
	#conclusion : code maa kei error xaina , dataset mai tyo word xaina 

def show_bar_plot():
	pass
# print(os.getcwd())
# files = get_files()
# print(len(files))
# words =read_file(files[100])
# words.sort()
# print(words)

words = get_all()

new_words,counts = count_words(words)

print(len(counts))
print(len(new_words))
# print(new_words)

#making a simple dataframe of the words and their counts
data = {"Words": new_words,
		"Occurances": counts}
df = pd.DataFrame(data)

print(df.head())
df.index += 1
print(df.head())

#save the file in excel
# df.to_excel("Word count.xlsx")

temp_df = df
temp_df = temp_df.sort_values("Occurances",ascending=False)
temp_df.reset_index(drop=True, inplace=True) 
temp_df.index += 1
print(temp_df.head(15))
# temp_df.to_excel("Word count.xlsx")

x_data = temp_df["Words"].values.tolist()
y_data = temp_df["Occurances"].values.tolist()
plt.title("Bar diagram of Top 20 Word Occurances")
plt.bar(x_data[0:20],y_data[0:20],width = 0.5)
plt.xlabel("Words")
plt.ylabel("Occurances")
plt.show()
