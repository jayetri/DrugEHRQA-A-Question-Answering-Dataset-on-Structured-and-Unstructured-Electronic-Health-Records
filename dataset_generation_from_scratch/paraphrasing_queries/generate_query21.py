import pandas as pd
import json
import numpy as np
import re

###########################Query 21  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/non_paraphrased_queries/query21.xlsx")   
print("count of samples:  " + str(len(df)))


for index, row in df.iterrows():
    result_for_hadm = re.search('LIST ALL THE MEDICINES AND THEIR DOSAGES PRESCRIBED TO THE PATIENT WITH ADMISSION ID = (.*) FOR', df.at[index, 'NL_Question'])
    hadm_value = result_for_hadm.group(1)
    problem_value = df.at[index, 'NL_Question'].split("FOR ",1)[1]
    
    if 1500 <= index <= 5184:
        df.at[index, 'NL_Question'] = "MAKE A LIST OF ALL THE DRUGS ALONG WITH THEIR DOSAGES TAKEN BY THE PATIENT SUFFERING FROM " + problem_value + ", HAVING AN ADMISSION ID OF " + hadm_value
    
    if 5185 <= index <= 7776:
        df.at[index, 'NL_Question'] = "WHAT ARE THE MEDICINES AND THEIR RESPECTIVE DOSAGES PRESCRIBED TO THE PATIENT WITH ADMISSION ID = " + hadm_value + " FOR " + problem_value
    
    if 7777 <= index <= 10368:  
        df.at[index, 'NL_Question'] = "FOR THE PATIENT WITH ADMISSION ID " + hadm_value + ", SUFFERING FROM " + problem_value + ", NAME ALL THE MEDICINES PRESCRIBED ALONG WITH THEIR RECOMMENDED DOSAGES"
'''
for index, row in df.iterrows():
    if 5 <= index <= 1200:
        df.at[index, 'NL_Question'] = "MAKE A LIST OF ALL THE DRUGS ALONG WITH THEIR DOSAGES TAKEN BY THE PATIENT SUFFERING FROM " + problem_value + ", HAVING AN ADMISSION ID OF " + hadm_value
    
    if 751 <= index <= 2000:
        df.at[index, 'NL_Question'] = "WHAT ARE THE MEDICINES AND THEIR RESPECTIVE DOSAGES PRESCRIBED TO THE PATIENT WITH ADMISSION ID = " + hadm_value + " FOR " + problem_value

    if 2001 <= index <= 2592:
        df.at[index, 'NL_Question'] = "FOR THE PATIENT WITH ADMISSION ID " + hadm_value + ", SUFFERING FROM " + problem_value + ", NAME ALL THE MEDICINES PRESCRIBED ALONG WITH THEIR RECOMMENDED DOSAGES"
'''

df = df.sample(frac=1)
df = df.sample(frac=1)
df = df.sample(frac=1)

df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation_big/paraphrased_queries/query21.xlsx", index = False, header=True)