import pandas as pd
import json
import numpy as np
import re

###########################Query 17  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/non_paraphrased_queries/query17.xlsx")   
print("count of samples:  " + str(len(df)))


for index, row in df.iterrows():
    result_for_hadm = re.search('WHY IS THE PATIENT WITH ADMISSION ID (.*) BEEN GIVEN', df.at[index, 'NL_Question'])
    hadm_value = result_for_hadm.group(1)
    drug_value = df.at[index, 'NL_Question'].split("GIVEN ",1)[1]    
    
    if 2517 <= index <= 5032:
        df.at[index, 'NL_Question'] = "MENTION THE REASON WHY THE PATIENT WITH AN ADMISSION ID OF " + hadm_value + " IS TAKING " + drug_value
    
    if 5033 <= index <= 7548:
        df.at[index, 'NL_Question'] = "WHAT IS THE REASON THE PATIENT WITH ADMISSION ID " + hadm_value + " IS ON " + drug_value
    
    if 7549 <= index <= 10067:  
        df.at[index, 'NL_Question'] = "WHY IS THE PATIENT WITH ADMISSION ID = " + hadm_value + " TAKING " + drug_value


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/paraphrased_queries/query17.xlsx", index = False, header=True)
