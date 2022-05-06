import pandas as pd
import json
import numpy as np



###########################Query 3  #####################################################
df = pd.read_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/non_paraphrased_queries/query3.xlsx")
print("count of query 3:  " + str(len(df)))

for index, row in df.iterrows():
    #print(index)
    if 127 <= index <= 252:
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("WHAT ARE THE LIST OF MEDICINES PRESCRIBED TO THE PATIENT WITH ADMISSION ID", "LIST THE MEDICINES RECOMMENDED TO BE TAKEN BY THE PATIENT HAVING AN ADMISSION ID")
    if 253 <= index <= 378:
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("WHAT ARE THE LIST OF MEDICINES PRESCRIBED TO THE PATIENT WITH ADMISSION ID", "WHAT ARE THE MEDICATIONS TAKEN BY THE PATIENT WITH ADMISSION ID")
    if 379 <= index <= 504:  
        df.at[index, 'NL_Question'] = df.at[index, 'NL_Question'].replace("WHAT ARE THE LIST OF MEDICINES PRESCRIBED TO THE PATIENT WITH ADMISSION ID", "NAME THE DRUGS PRESCRIBED TO THE PATIENT WITH ADMISSION ID =")
df = df.sample(frac=1)
df.to_excel("/home/jayetri/redo_drug_dataset/multimodal_dataset_generation/paraphrased_queries/query3.xlsx", index = False, header=True)





