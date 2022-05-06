import pandas as pd
import json
import numpy as np



###########################Query 4  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/non_paraphrased_queries/query4.xlsx") 
print("count of samples:  " + str(len(df)))

for index, row in df.iterrows():
    #print(index)
    if 2331 <= index <= 4660:
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("WHAT IS THE DRUG STRENGTH OF", "WHAT IS THE STRENGTH OF THE MEDICINE")
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("PRESCRIBED TO THE PATIENT WITH ADMISSION ID", "TAKEN BY THE PATIENT HAVING AN ADMISSION ID =")
    
    if 4661 <= index <= 6990:
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("WHAT IS THE DRUG STRENGTH OF", "WHAT IS THE DRUG INTENSITY OF")
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("PRESCRIBED TO THE PATIENT WITH ADMISSION ID", "GIVEN TO THE PATIENT WITH AN ADMISSION ID =")
    
    
    if 6991 <= index <= 9320:  
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("WHAT IS THE DRUG STRENGTH OF", "MENTION THE DRUG INTENSITY OF")
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("PRESCRIBED TO THE PATIENT WITH ADMISSION ID", "PRESCRIBED TO THE PATIENT WHOSE ADMISSION ID IS")


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/paraphrased_queries/query4.xlsx", index = False, header=True)
