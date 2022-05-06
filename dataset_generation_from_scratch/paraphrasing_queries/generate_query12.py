import pandas as pd
import json
import numpy as np
import re

###########################Query 12  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/non_paraphrased_queries/query12.xlsx")  
print("count of samples:  " + str(len(df)))


for index, row in df.iterrows():
    word_list = df.at[index, 'NL_Question'].split()
    hadm_value = word_list[-1]
    result = re.search('WHAT IS THE DOSAGE OF (.*) PRESCRIBED TO THE PATIENT WITH ADMISSION ID', df.at[index, 'NL_Question'])
    drug_value = result.group(1)
    
    
    if 1921 <= index <= 3840:
        df.at[index, 'NL_Question'] = "WHAT IS THE DOSE OF " + drug_value + " THAT THE PATIENT WITH ADMISSION ID = " + hadm_value + " HAS BEEN PRESCRIBED"
    
    if 3841 <= index <= 5760:
        df.at[index, 'NL_Question'] = "WHAT WAS THE DOSAGE OF " + drug_value + " THAT WAS PRESCRIBED TO THE PATIENT WITH ADMISSION ID = " + hadm_value
    
    if 5761 <= index <= 7680:  
        df.at[index, 'NL_Question'] = "SPECIFY THE DOSE OF THE MEDICINE " + drug_value + " PRESCRIBED TO THE PATIENT WITH ADMISSION ID = " + hadm_value


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/paraphrased_queries/query12.xlsx", index = False, header=True)
