import pandas as pd
import json
import numpy as np
import re

###########################Query 19  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/non_paraphrased_queries/query19.xlsx")   
print("count of samples:  " + str(len(df)))


for index, row in df.iterrows():
    result_for_hadm = re.search('WHAT IS THE MEDICATION PRESCRIBED TO THE PATIENT WITH ADMISSION ID (.*) FOR', df.at[index, 'NL_Question'])
    hadm_value = result_for_hadm.group(1)
    problem_value = df.at[index, 'NL_Question'].split("FOR ",1)[1]    
    
    if 3373 <= index <= 6744:
        df.at[index, 'NL_Question'] = "WHICH MEDICINES ARE TAKEN BY THE PATIENT SUFFERING FROM " + problem_value + " HAVING AN ADMISSION ID OF " + hadm_value
    
    if 6745 <= index <= 10116:
        df.at[index, 'NL_Question'] = "WHAT MEDICATION IS THE PATIENT WITH AN ADMISSION ID OF " + hadm_value + " TAKING FOR " + problem_value
    
    if 10117 <= index <= 13491:  
        df.at[index, 'NL_Question'] = "FOR " + problem_value + ", NAME THE DRUGS THAT HAS BEEN RECOMMENDED TO BE TAKEN BY THE PATIENT WITH ADMISSION ID = " + hadm_value


df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/paraphrased_queries/query19.xlsx", index = False, header=True)
