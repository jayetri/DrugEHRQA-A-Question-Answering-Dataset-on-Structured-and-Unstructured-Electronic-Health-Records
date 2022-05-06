import pandas as pd
import json
import numpy as np
import re



###########################Query 8  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/non_paraphrased_queries/query8.xlsx") 
print("count of samples:  " + str(len(df)))

for index, row in df.iterrows():
    word_list = df.at[index, 'NL_Question'].split()
    hadm_value = word_list[-1]
    #print("index is:", index)
    #print(word_list)
    #print(hadm_value)
    result = re.search('WHAT IS THE FORM OF (.*) PRESCRIBED TO THE PATIENT WITH ADMISSION ID', df.at[index, 'NL_Question'])
    drug_value = result.group(1)
    #print(drug_value)

    if 2100 <= index <= 2300:
        df.at[index, 'NL_Question'] = "PLEASE SPECIFY IF THE MEDICINE " + drug_value + " PRESCRIBED TO THE PATIENT WITH ADMISSION ID " + hadm_value + " IS IN THE FORM OF TABLET, CAPSULE, SYRUP OR IN ANY OTHER FORM"
        print(df.at[index, 'NL_Question'])
    
    if 3500 <= index <= 5508:
        df.at[index, 'NL_Question'] = "MENTION THE FORM OF " + drug_value + " TAKEN BY THE PATIENT HAVING AN ADMISSION ID = " + hadm_value
        print(df.at[index, 'NL_Question'])
    
    if 5509 <= index <= 7344:  
        df.at[index, 'NL_Question'] = "IN WHAT FORM WAS " + drug_value + " GIVEN TO THE PATIENT WITH ADMISSION ID " + hadm_value
        print(df.at[index, 'NL_Question'])


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/paraphrased_queries/query8.xlsx", index = False, header=True)
