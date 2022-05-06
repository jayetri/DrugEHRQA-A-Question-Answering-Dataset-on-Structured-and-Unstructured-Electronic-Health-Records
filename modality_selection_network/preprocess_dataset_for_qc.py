import pandas as pd
import numpy as np
import json
import re


###########################Query 3 start #####################################################
print("Start Query3----------------------------------------------------------------------------------->")
df_q3 = pd.read_excel ('paraphrased_queries/query3.xlsx')
print("No of queries in q3 before dropping blank cells: ", len(df_q3))  
df_q3 = df_q3.dropna()
df_q3['qc_label'] = '1'
df_q3['template_id'] = '3'
print("No of queries in q3 after dropping blank cells: ", len(df_q3))
df_q3.to_excel("added_label/query3.xlsx", index = False,  header=True)


###########################Query 4 start #####################################################
print("Start Query4----------------------------------------------------------------------------------->")
df_q4 = pd.read_excel ('paraphrased_queries/query4.xlsx')
print("No of queries in q4 before dropping blank cells: ", len(df_q4))  
df_q4 = df_q4.dropna()
df_q4['qc_label'] = '1'
df_q4['template_id'] = '4'
print("No of queries in q4 after dropping blank cells: ", len(df_q4))
df_q4.to_excel("added_label/query4.xlsx", index = False,  header=True)

###########################Query 8 start #####################################################
print("Start Query8----------------------------------------------------------------------------------->")
df_q8 = pd.read_excel ('paraphrased_queries/query8.xlsx')
print("No of queries in q8 before dropping blank cells: ", len(df_q8))  
df_q8 = df_q8.dropna()
df_q8['qc_label'] = '1'
df_q8['template_id'] = '8'
print("No of queries in q8 after dropping blank cells: ", len(df_q8))
df_q8.to_excel("added_label/query8.xlsx", index = False,  header=True)

###########################Query 10 start #####################################################
print("Start Query10----------------------------------------------------------------------------------->")
df_q10 = pd.read_excel ('paraphrased_queries/query10.xlsx')
print("No of queries in q10 before dropping blank cells: ", len(df_q10))  
df_q10 = df_q10.dropna()
df_q10['qc_label'] = '1'
df_q10['template_id'] = '10'
print("No of queries in q10 after dropping blank cells: ", len(df_q10))
df_q10.to_excel("added_label/query10.xlsx", index = False,  header=True)

###########################Query 12 start #####################################################
print("Start Query12----------------------------------------------------------------------------------->")
df_q12 = pd.read_excel ('paraphrased_queries/query12.xlsx')
print("No of queries in q12 before dropping blank cells: ", len(df_q12))  
df_q12 = df_q12.dropna()
df_q12['qc_label'] = '1'
df_q12['template_id'] = '12'
print("No of queries in q12 after dropping blank cells: ", len(df_q12))
df_q12.to_excel("added_label/query12.xlsx", index = False,  header=True)

###########################Query 13 start #####################################################
print("Start Query13----------------------------------------------------------------------------------->")
df_q13 = pd.read_excel ('paraphrased_queries/query13.xlsx')
print("No of queries in q13 before dropping blank cells: ", len(df_q13))  
df_q13 = df_q13.dropna()
df_q13['qc_label'] = '0'
df_q13['template_id'] = '13'
print("No of queries in q13 after dropping blank cells: ", len(df_q13))
df_q13.to_excel("added_label/query13.xlsx", index = False,  header=True)

###########################Query 17 start #####################################################
print("Start Query17----------------------------------------------------------------------------------->")
df_q17 = pd.read_excel ('paraphrased_queries/query17.xlsx')
print("No of queries in q17 before dropping blank cells: ", len(df_q17))  
df_q17 = df_q17.dropna()
df_q17['qc_label'] = '0'
df_q17['template_id'] = '17'
print("No of queries in q17 after dropping blank cells: ", len(df_q17))
df_q17.to_excel("added_label/query17.xlsx", index = False,  header=True)

###########################Query 19 start #####################################################
print("Start Query19----------------------------------------------------------------------------------->")
df_q19 = pd.read_excel ('paraphrased_queries/query19.xlsx')
print("No of queries in q19 before dropping blank cells: ", len(df_q19))  
df_q19 = df_q19.dropna()
df_q19['qc_label'] = '0'
df_q19['template_id'] = '19'
print("No of queries in q19 after dropping blank cells: ", len(df_q19))
df_q19.to_excel("added_label/query19.xlsx", index = False,  header=True)


'''
###########################Query 21 start #####################################################
print("Start Query21----------------------------------------------------------------------------------->")
df_q21 = pd.read_excel ('paraphrased_queries/query21.xlsx')
print("No of queries in q21 before dropping blank cells: ", len(df_q21))  
df_q21 = df_q21.dropna()
df_q21['qc_label'] = '0'
df_q21['template_id'] = '21'
print("No of queries in q21 after dropping blank cells: ", len(df_q21))
df_q21.to_excel("added_label/query21.xlsx", index = False,  header=True)

'''

'''

final_concat_df = pd.concat([df_q3, df_q4, df_q8, df_q10, df_q12, df_q13, df_q17, df_q19], axis=0)
print("Total number of queries: ", len(final_concat_df))  
final_concat_df = final_concat_df.sample(frac=1)
final_concat_df.to_excel("added_label/all_queries.xlsx", index = False,  header=True)

msk = np.random.rand(len(final_concat_df)) < 0.9
train_df = final_concat_df[msk]
test_df = final_concat_df[~msk]
train_df.to_excel("added_label/train.xlsx", index = False,  header=True)
test_df.to_excel("added_label/test.xlsx", index = False,  header=True)
train_df.to_csv("added_label/train.csv", index = False,  header=True)
test_df.to_csv("added_label/test.csv", index = False,  header=True)
'''

print("Total number of queries in train: ", len(train_df))  
print("Total number of queries in test: ", len(test_df))  

