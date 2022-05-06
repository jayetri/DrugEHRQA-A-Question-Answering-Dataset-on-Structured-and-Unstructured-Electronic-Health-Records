import pandas as pd
import json
import numpy as np
import re

###########################Query 10  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/non_paraphrased_queries/query10.xlsx")  
print("count of samples:  " + str(len(df)))

for index, row in df.iterrows():
    word_list = df.at[index, 'NL_Question'].split()
    hadm_value = word_list[-1]
    result = re.search('WHAT IS THE ROUTE OF ADMINISTRATION OF THE DRUG (.*) FOR PATIENT WITH ADMISSION ID', df.at[index, 'NL_Question'])
    drug_value = result.group(1)
    

    if 2133 <= index <= 4264:
        df.at[index, 'NL_Question'] = "WHAT SHOULD BE THE MODE OF ENTRY OF THE DRUG " + drug_value + " INTO THE BODY OF THE PATIENT HAVING AN ADMISSION ID = " + hadm_value
        #print(df.at[index, 'NL_Question'])
    
    if 4265 <= index <= 6396:
        df.at[index, 'NL_Question'] = "FOR THE PATIENT HAVING AN ADMISSION ID = " + hadm_value + ", WHAT IS THE RECOMMENDED ROUTE OF DRUG ADMINISTRATION FOR " + drug_value
        #print(df.at[index, 'NL_Question'])
    
    if 6397 <= index <= 8530:  
        df.at[index, 'NL_Question'] = "MENTION THE ROUTE OF ADMINISTRATION FOR THE MEDICINE " + drug_value + " RECOMMENDED TO THE PATIENT WITH ADMISSION ID = " + hadm_value
        #print(df.at[index, 'NL_Question'])


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/paraphrased_queries/query10.xlsx", index = False, header=True)
