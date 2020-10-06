import os
import glob
import sys
import re
import shutil as sh
# import pyperclip

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
# AUDIO = "audio"

MODES =[AUDIBLE,WHISPERED,SILENT] 


file_re = re.compile(r"\d\d\d-\d\d\d-\d\d\d\d")

# file_re = re.compile(r"emg_\d\d\d-\d\d\d-\d\d\d\d")

# print(os.getcwd())

os.chdir(MAIN_DIR)

def find_files(path=CATEGORY):
	"""
	function that finds files of all categories and saves them in a corresponding list
	"""
	categories = []
	for file in glob.glob("Subsets/*"):
		categories.append(file)

	# print(categories)

	all_files = {}
	for cat in categories:
		with open(cat,"r") as file:
			all_files[cat.split("/")[-1]] = file_re.findall(file.read())
			# print(cat.split("/")[-1])

	return all_files

def arrange_files(session=False):
	all_files = find_files()
	files = []
	# print(all_files.keys())
	
	for keys in list(all_files.keys()):
		#for EMG data
		for file in glob.glob("emg/*/*/*.adc"):
			match = file_re.search(file.replace("_","-")).group()
			if(match in all_files[keys]):
				# code to copy files to respective folders
				dest = os.path.join(MAIN_DIR,DATASET,keys.split(".")[-1],EMG,match+".adc")
				sh.copyfile(file,dest)
		
		#for audio data 
		for file in glob.glob("audio/*/*/*.wav"):
			match = file_re.search(file.replace("_","-")).group()
			if(match in all_files[keys]):
				# code to copy files to respective folders
				dest = os.path.join(MAIN_DIR,DATASET,keys.split(".")[-1],AUDIO,match+".wav")
				sh.copyfile(file,dest)

	return

def make_tree():
	dataset = [AUDIBLE,SILENT,WHISPERED]
	for data in dataset:
		os.makedirs(os.path.join(MAIN_DIR,DATASET,data,AUDIO),exist_ok=True)
		os.makedirs(os.path.join(MAIN_DIR,DATASET,data,EMG),exist_ok=True)

def main():
	make_tree()
	arrange_files()


if __name__ == '__main__':
	main()