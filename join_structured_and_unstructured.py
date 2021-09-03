import pandas as pd
import numpy as np
import json
import re


###########################Query 1 start #####################################################
print("Start Query1----------------------------------------------------------------------------------->")
df_q1_struct = pd.DataFrame(pd.read_csv("structured_queries/query1.csv"))
df_q1_unstruct = pd.read_excel ('unstructured_queries/query1.xlsx')
merged_q1 = pd.merge(df_q1_struct,df_q1_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q1["_merge"]   

merged_q1.to_excel("merged_queries/query1.xlsx", index = False,  header=True)
print("No of queries in merged query1: ", len(merged_q1))


###########################Query 2 start #####################################################
print("Start Query2----------------------------------------------------------------------------------->")
df_q2_struct = pd.DataFrame(pd.read_csv("structured_queries/query2.csv"))
df_q2_unstruct = pd.read_excel ('unstructured_queries/query2.xlsx')
merged_q2 = pd.merge(df_q2_struct,df_q2_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q2["_merge"]   

merged_q2.to_excel("merged_queries/query2.xlsx", index = False,  header=True)
print("No of queries in merged query2: ", len(merged_q2))

###########################Query 3 start #####################################################
print("Start Query3----------------------------------------------------------------------------------->")
df_q3_struct = pd.DataFrame(pd.read_csv("structured_queries/query3.csv"))
df_q3_unstruct = pd.read_excel ('unstructured_queries/query3.xlsx')
merged_q3 = pd.merge(df_q3_struct,df_q3_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q3["_merge"]   

merged_q3.to_excel("merged_queries/query3.xlsx", index = False,  header=True)
print("No of queries in merged query3: ", len(merged_q3))

###########################Query 4 start #####################################################
print("Start Query4----------------------------------------------------------------------------------->")
df_q4_struct = pd.DataFrame(pd.read_csv("structured_queries/query4.csv"))
df_q4_unstruct = pd.read_excel ('unstructured_queries/query4.xlsx')
merged_q4 = pd.merge(df_q4_struct,df_q4_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q4["_merge"]   

merged_q4.to_excel("merged_queries/query4.xlsx", index = False,  header=True)
print("No of queries in merged query4: ", len(merged_q4))

###########################Query 5 start #####################################################
print("Start Query5----------------------------------------------------------------------------------->")
df_q5_struct = pd.DataFrame(pd.read_csv("structured_queries/query5.csv"))
df_q5_unstruct = pd.read_excel ('unstructured_queries/query5.xlsx')
merged_q5 = pd.merge(df_q5_struct,df_q5_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q5["_merge"]   

merged_q5.to_excel("merged_queries/query5.xlsx", index = False,  header=True)
print("No of queries in merged query5: ", len(merged_q5))

###########################Query 6 start #####################################################
print("Start Query6----------------------------------------------------------------------------------->")
df_q6_struct = pd.DataFrame(pd.read_csv("structured_queries/query6.csv"))
df_q6_unstruct = pd.read_excel ('unstructured_queries/query6.xlsx')
merged_q6 = pd.merge(df_q6_struct,df_q6_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q6["_merge"]   

merged_q6.to_excel("merged_queries/query6.xlsx", index = False,  header=True)
print("No of queries in merged query6: ", len(merged_q6))

###########################Query 7 start #####################################################
print("Start Query7----------------------------------------------------------------------------------->")
df_q7_struct = pd.DataFrame(pd.read_csv("structured_queries/query7.csv"))
df_q7_unstruct = pd.read_excel ('unstructured_queries/query7.xlsx')
merged_q7 = pd.merge(df_q7_struct,df_q7_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q7["_merge"]   

merged_q7.to_excel("merged_queries/query7.xlsx", index = False,  header=True)
print("No of queries in merged query7: ", len(merged_q7))

###########################Query 8 start #####################################################
print("Start Query8----------------------------------------------------------------------------------->")
df_q8_struct = pd.DataFrame(pd.read_csv("structured_queries/query8.csv"))
df_q8_unstruct = pd.read_excel ('unstructured_queries/query8.xlsx')
merged_q8 = pd.merge(df_q8_struct,df_q8_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q8["_merge"]   

merged_q8.to_excel("merged_queries/query8.xlsx", index = False,  header=True)
print("No of queries in merged query8: ", len(merged_q8))

###########################Query 9 start #####################################################
print("Start Query9----------------------------------------------------------------------------------->")
df_q9_struct = pd.DataFrame(pd.read_csv("structured_queries/query9.csv"))
df_q9_unstruct = pd.read_excel ('unstructured_queries/query9.xlsx')
merged_q9 = pd.merge(df_q9_struct,df_q9_unstruct,on='NL_Question',how='outer',indicator=True)
del merged_q9["_merge"]   

merged_q9.to_excel("merged_queries/query9.xlsx", index = False,  header=True)
print("No of queries in merged query9: ", len(merged_q9))

Total_merged_queries = len(merged_q1) + len(merged_q2) + len(merged_q3) + len(merged_q4) + len(merged_q5) + len(merged_q6) + len(merged_q7) + len(merged_q8) + len(merged_q9)
print("Total_merged_queries:", Total_merged_queries)  
