import pandas as pd
import json
import numpy as np
import re

###########################Query 13  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/non_paraphrased_queries/query13.xlsx")  
print("count of samples:  " + str(len(df)))


for index, row in df.iterrows():
    result_for_hadm = re.search('HOW LONG HAS THE PATIENT WITH ADMISSION ID = (.*) BEEN TAKING', df.at[index, 'NL_Question'])
    hadm_value = result_for_hadm.group(1)
    drug_value = df.at[index, 'NL_Question'].split("TAKING ",1)[1]
    
    
    if 500 <= index <= 1500:
        df.at[index, 'NL_Question'] = "MENTION THE TOTAL DURATION DURING WHICH THE PATIENT WITH ADMISSION ID = " + hadm_value + " HAS BEEN TAKING " + drug_value
    
    if 2293 <= index <= 3438:
        df.at[index, 'NL_Question'] = "HOW LONG HAS THE PATIENT WHOSE ADMISSION ID IS " + hadm_value + " BEEN ON " + drug_value
    
    if 3439 <= index <= 4584:  
        df.at[index, 'NL_Question'] = "WHAT IS THE PRESCRIBED DURATION DURING WHICH THE PATIENT WITH ADMISSION ID " + hadm_value + " IS SUPPOSED TO TAKE " + drug_value


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/paraphrased_queries/query13.xlsx", index = False, header=True)
