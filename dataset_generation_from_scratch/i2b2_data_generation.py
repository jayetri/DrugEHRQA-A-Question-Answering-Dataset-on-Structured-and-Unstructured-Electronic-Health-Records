import pandas as pd
import glob
import os

# Training dataset #

def create_training_set():
	globbed_files = glob.glob("/home/jayetri/redo_drug_dataset/i2b2 data/all_files/training_20180910/*.ann")
	data_train = []

	for csv in globbed_files:
		print(csv)
		frame = pd.read_csv(csv, sep='\t', header=None)
		filename = os.path.basename(csv)
		frame['hadm_id'] = filename[:6]
		#frame.columns = frame.columns.str.replace(';', '')
		frame = frame.replace(';',' ', regex=True)
		data_train.append(frame)

	print("generating train dataset")
	bigframe_train = pd.concat(data_train, ignore_index=True) #dont want pandas to try an align row indexes
	bigframe_train.to_csv("/home/jayetri/redo_drug_dataset/i2b2 data/training_data.csv")
	return data_train


def create_test_set():

	globbed_files = glob.glob("/home/jayetri/redo_drug_dataset/i2b2 data/all_files/test/*.ann")
	data_test = []

	for csv in globbed_files:
		print(csv)
		frame = pd.read_csv(csv, sep='\t', header=None)
		filename = os.path.basename(csv)
		frame['hadm_id'] = filename[:6]
		frame = frame.replace(';',' ', regex=True)
		data_test.append(frame)

	print("generating test dataset")
	bigframe_test = pd.concat(data_test, ignore_index=True) #dont want pandas to try an align row indexes
	bigframe_test.to_csv("/home/jayetri/redo_drug_dataset/i2b2 data/test_data.csv")
	return data_test

# Combine Datasets #


def combine_i2b2():
	create_training_set()
	create_test_set()
	train_df = pd.read_csv("/home/jayetri/redo_drug_dataset/i2b2 data/training_data.csv")
	test_df = pd.read_csv("/home/jayetri/redo_drug_dataset/i2b2 data/test_data.csv")

	dataset = []
	
	#train_df = pd.DataFrame(train,columns=['index','category','desc','name','hadm_id'])
	dataset.append(train_df)
	dataset.append(test_df)
	
	print("generating final dataset")
	bigframe = pd.concat(dataset, ignore_index=True) #dont want pandas to try an align row indexes
	#bigframe.to_csv("i2b2_data.csv")
	bigframe.rename(columns = {"2": "value"}, inplace = True)
	bigframe.to_excel("/home/jayetri/redo_drug_dataset/i2b2_data.xlsx", index = False,  header=True)



combine_i2b2()
