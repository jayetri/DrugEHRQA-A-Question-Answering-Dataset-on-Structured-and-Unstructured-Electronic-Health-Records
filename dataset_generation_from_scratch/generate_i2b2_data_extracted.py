import pandas as pd
import json
import numpy as np



'''
df_q3 = pd.read_csv ("/home/jayetri/redo_drug_dataset/i2b2_data_extracted.csv")
print("count of query 19 before dropping:  " + str(len(df_q3)))
for index, row in df_q3.iterrows():
    if index == 133116:
        print(index)
        print( df_q3.at[index,'Row_id'], df_q3.at[index,'value'] , df_q3.at[index,'hadm_id'] )
'''

#df = pd.read_csv ("/home/jayetri/redo_drug_dataset/i2b2_data.csv")
df = pd.read_excel("/home/jayetri/redo_drug_dataset/i2b2_data.xlsx")
for index, row in df.iterrows():
    if index % 1000 == 0:
        print(index)
    #print("Index is: ", index)
    #print(df.at[index, '1'].split())
    df.at[index, 'type'] = df.at[index, '1'].split()[0]

    if df.at[index, '0'].startswith('T'):
        df.at[index, 'arg1_value'] = ''
        df.at[index, 'arg2_value'] = ''
    else:
        arg1 = df.at[index, '1'].split()[1].split(':')[1]
        hadm_id = df.at[index, 'hadm_id']
        #print(arg1, hadm_id)
        arg1_value_dataframe = df[(df["0"] == arg1) & (df["hadm_id"] == hadm_id)]
        #print("arg1 is: ",  arg1_value_dataframe["value"].values[0])
        df.at[index, 'arg1_value'] = arg1_value_dataframe["value"].values[0]


        arg2 = df.at[index, '1'].split()[2].split(':')[1]
        hadm_id = df.at[index, 'hadm_id']
        #print(arg2, hadm_id)
        arg2_value_dataframe = df[(df["0"] == arg2) & (df["hadm_id"] == hadm_id)]
        #print("arg2 is: ",  arg2_value_dataframe["value"].values[0])
        df.at[index, 'arg2_value'] = arg2_value_dataframe["value"].values[0]
        #print("Done---------------------------------------------------------------------------------------")

df.to_excel("/home/jayetri/redo_drug_dataset/i2b2_data_extracted.xlsx", index = False,  header=True)

