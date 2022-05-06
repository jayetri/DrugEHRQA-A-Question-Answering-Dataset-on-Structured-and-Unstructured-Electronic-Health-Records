import io
import random
import numpy as np
import pandas as pd
import re

'''
Results: 
Number of filtered questions is:  716
Number of elements in q3 test set before filtering 0
Number of elements in q3 test set 0
Number of elements in q4 test set before filtering 200
Number of elements in q4 test set 200
Number of elements in q8 test set before filtering 50
Number of elements in q8 test set 50
Number of elements in q10 test set before filtering 50
Number of elements in q10 test set 50
Number of elements in q12 test set before filtering 232
Number of elements in q12 test set 232
Number of elements in q13 test set before filtering 28
Number of elements in q13 test set 28
Number of elements in q17 test set before filtering 150
Number of elements in q17 test set 150
Number of elements in q19 test set before filtering 12
Number of elements in q19 test set 6
Total elements in the test before filtering:  722
No of rows in df_final before filtering  722
list_of_questions_before_filtered_in_test: 722
Total elements in the test after filtering:  716
No of rows in df_final after filtering  716
list_of_questions_after_filtered_in_test: 716
Number of elements in q3 train set 475
Number of elements in q4 train set 2750
Number of elements in q8 train set 2175
Number of elements in q10 train set 2763
Number of elements in q12 train set 2092
Number of elements in q13 train set 249
Number of elements in q17 train set 1358
Number of elements in q19 train set 114
Total elements in the train set after filtering:  11976
No of rows in df_final  11976
list_of_questions_filtered_in_train: 11976


'''

with open('correct_list_file.txt') as f:
    lines = [line.rstrip() for line in f]


for i in range(len(lines)):
    lines[i] = lines[i].upper()
    lines[i] = re.sub(' ([-]) ?',r'\1', lines[i])
    lines[i] = lines[i].replace("( ", "(")
    lines[i] = lines[i].replace(" )", ")")
    #print(lines[i])

print("Number of filtered questions is: ", len(lines))   ###There are 1129 out of 1280 questions filtered.


######Filter Test set
df = pd.read_csv ('test.csv')
df_test_q3 = df[df['template_id'] == 3] 
del df_test_q3['template_id']
del df_test_q3['qc_label'] 
del df_test_q3['intersecting_answers'] 
print("Number of elements in q3 test set before filtering", len(df_test_q3))
df_test_q3.to_excel("non_intersecting_answers/test_set_not_filtered/query3.xlsx", index = False,  header=True)
df_test_q3_filtered = df_test_q3[df_test_q3['NL_Question'].isin(lines)] 
print("Number of elements in q3 test set", len(df_test_q3_filtered))
df_test_q3_filtered.to_excel("non_intersecting_answers/test_set/query3.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q4 = df[df['template_id'] == 4] 
del df_test_q4['template_id']
del df_test_q4['qc_label'] 
del df_test_q4['intersecting_answers'] 
print("Number of elements in q4 test set before filtering", len(df_test_q4))
df_test_q4.to_excel("non_intersecting_answers/test_set_not_filtered/query4.xlsx", index = False,  header=True)
df_test_q4_filtered = df_test_q4[df_test_q4['NL_Question'].isin(lines)] 
print("Number of elements in q4 test set", len(df_test_q4_filtered))
df_test_q4_filtered.to_excel("non_intersecting_answers/test_set/query4.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q8 = df[df['template_id'] == 8] 
del df_test_q8['template_id']
del df_test_q8['qc_label'] 
del df_test_q8['intersecting_answers'] 
print("Number of elements in q8 test set before filtering", len(df_test_q8))
df_test_q8.to_excel("non_intersecting_answers/test_set_not_filtered/query8.xlsx", index = False,  header=True)
df_test_q8_filtered = df_test_q8[df_test_q8['NL_Question'].isin(lines)] 
print("Number of elements in q8 test set", len(df_test_q8_filtered))
df_test_q8_filtered.to_excel("non_intersecting_answers/test_set/query8.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q10 = df[df['template_id'] == 10] 
del df_test_q10['template_id']
del df_test_q10['qc_label'] 
del df_test_q10['intersecting_answers'] 
print("Number of elements in q10 test set before filtering", len(df_test_q10))
df_test_q10.to_excel("non_intersecting_answers/test_set_not_filtered/query10.xlsx", index = False,  header=True)
df_test_q10_filtered = df_test_q10[df_test_q10['NL_Question'].isin(lines)] 
print("Number of elements in q10 test set", len(df_test_q10_filtered))
df_test_q10_filtered.to_excel("non_intersecting_answers/test_set/query10.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q12 = df[df['template_id'] == 12] 
del df_test_q12['template_id']
del df_test_q12['qc_label'] 
del df_test_q12['intersecting_answers'] 
print("Number of elements in q12 test set before filtering", len(df_test_q12))
df_test_q12.to_excel("non_intersecting_answers/test_set_not_filtered/query12.xlsx", index = False,  header=True)
df_test_q12_filtered = df_test_q12[df_test_q12['NL_Question'].isin(lines)] 
print("Number of elements in q12 test set", len(df_test_q12_filtered))
df_test_q12_filtered.to_excel("non_intersecting_answers/test_set/query12.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q13 = df[df['template_id'] == 13] 
del df_test_q13['template_id']
del df_test_q13['qc_label'] 
del df_test_q13['intersecting_answers'] 
print("Number of elements in q13 test set before filtering", len(df_test_q13))
df_test_q13.to_excel("non_intersecting_answers/test_set_not_filtered/query13.xlsx", index = False,  header=True)
df_test_q13_filtered = df_test_q13[df_test_q13['NL_Question'].isin(lines)] 
print("Number of elements in q13 test set", len(df_test_q13_filtered))
df_test_q13_filtered.to_excel("non_intersecting_answers/test_set/query13.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q17 = df[df['template_id'] == 17] 
del df_test_q17['template_id']
del df_test_q17['qc_label'] 
del df_test_q17['intersecting_answers'] 
print("Number of elements in q17 test set before filtering", len(df_test_q17))
df_test_q17.to_excel("non_intersecting_answers/test_set_not_filtered/query17.xlsx", index = False,  header=True)
df_test_q17_filtered = df_test_q17[df_test_q17['NL_Question'].isin(lines)] 
print("Number of elements in q17 test set", len(df_test_q17_filtered))
df_test_q17_filtered.to_excel("non_intersecting_answers/test_set/query17.xlsx", index = False,  header=True)


df = pd.read_csv ('test.csv')
df_test_q19 = df[df['template_id'] == 19] 
del df_test_q19['template_id']
del df_test_q19['qc_label'] 
del df_test_q19['intersecting_answers'] 
print("Number of elements in q19 test set before filtering", len(df_test_q19))
df_test_q19.to_excel("non_intersecting_answers/test_set_not_filtered/query19.xlsx", index = False,  header=True)
df_test_q19_filtered = df_test_q19[df_test_q19['NL_Question'].isin(lines)] 
print("Number of elements in q19 test set", len(df_test_q19_filtered))
df_test_q19_filtered.to_excel("non_intersecting_answers/test_set/query19.xlsx", index = False,  header=True)


print("Total elements in the test before filtering: ", len(df_test_q3) + len(df_test_q4) + len(df_test_q8) + len(df_test_q10) + len(df_test_q12) + len(df_test_q13) + len(df_test_q17) + len(df_test_q19))

df_final_test = pd.concat([df_test_q3, df_test_q4 , df_test_q8 , df_test_q10 , df_test_q12 , df_test_q13 , df_test_q17 , df_test_q19], ignore_index=True)
print("No of rows in df_final before filtering ", len(df_final_test))

list_of_questions_before_filtered_in_test = df_final_test['NL_Question'].tolist()
print("list_of_questions_before_filtered_in_test:", len(list_of_questions_before_filtered_in_test))





print("Total elements in the test after filtering: ", len(df_test_q3_filtered) + len(df_test_q4_filtered) + len(df_test_q8_filtered) + len(df_test_q10_filtered) + len(df_test_q12_filtered) + len(df_test_q13_filtered) + len(df_test_q17_filtered) + len(df_test_q19_filtered))

df_final_test_filtered = pd.concat([df_test_q3_filtered, df_test_q4_filtered , df_test_q8_filtered , df_test_q10_filtered , df_test_q12_filtered , df_test_q13_filtered , df_test_q17_filtered , df_test_q19_filtered], ignore_index=True)
print("No of rows in df_final after filtering ", len(df_final_test_filtered))

list_of_questions_after_filtered_in_test = df_final_test_filtered['NL_Question'].tolist()
print("list_of_questions_after_filtered_in_test:", len(list_of_questions_after_filtered_in_test))

######Train set
df = pd.read_csv ('train.csv')
df_train_q3 = df[df['template_id'] == 3] 
del df_train_q3['template_id']
del df_train_q3['qc_label'] 
del df_train_q3['intersecting_answers'] 
print("Number of elements in q3 train set", len(df_train_q3))
df_train_q3.to_excel("non_intersecting_answers/train_and_dev_set/query3.xlsx", index = False,  header=True)


df_train_q4 = df[df['template_id'] == 4] 
del df_train_q4['template_id']
del df_train_q4['qc_label']  
del df_train_q4['intersecting_answers'] 
print("Number of elements in q4 train set", len(df_train_q4))
df_train_q4.to_excel("non_intersecting_answers/train_and_dev_set/query4.xlsx", index = False,  header=True)


df_train_q8 = df[df['template_id'] == 8] 
del df_train_q8['template_id']
del df_train_q8['qc_label']  
del df_train_q8['intersecting_answers'] 
print("Number of elements in q8 train set", len(df_train_q8))
df_train_q8.to_excel("non_intersecting_answers/train_and_dev_set/query8.xlsx", index = False,  header=True)


df_train_q10 = df[df['template_id'] == 10] 
del df_train_q10['template_id']
del df_train_q10['qc_label']  
del df_train_q10['intersecting_answers'] 
print("Number of elements in q10 train set", len(df_train_q10))
df_train_q10.to_excel("non_intersecting_answers/train_and_dev_set/query10.xlsx", index = False,  header=True)


df_train_q12 = df[df['template_id'] == 12] 
del df_train_q12['template_id']
del df_train_q12['qc_label']  
del df_train_q12['intersecting_answers'] 
print("Number of elements in q12 train set", len(df_train_q12))
df_train_q12.to_excel("non_intersecting_answers/train_and_dev_set/query12.xlsx", index = False,  header=True)


df_train_q13 = df[df['template_id'] == 13] 
del df_train_q13['template_id']
del df_train_q13['qc_label'] 
del df_train_q13['intersecting_answers'] 
print("Number of elements in q13 train set", len(df_train_q13))
df_train_q13.to_excel("non_intersecting_answers/train_and_dev_set/query13.xlsx", index = False,  header=True)


df_train_q17 = df[df['template_id'] == 17] 
del df_train_q17['template_id']
del df_train_q17['qc_label'] 
del df_train_q17['intersecting_answers'] 
print("Number of elements in q17 train set", len(df_train_q17))
df_train_q17.to_excel("non_intersecting_answers/train_and_dev_set/query17.xlsx", index = False,  header=True)


df_train_q19 = df[df['template_id'] == 19] 
del df_train_q19['template_id']
del df_train_q19['qc_label']
del df_train_q19['intersecting_answers'] 
print("Number of elements in q19 train set", len(df_train_q19))
df_train_q19.to_excel("non_intersecting_answers/train_and_dev_set/query19.xlsx", index = False,  header=True)


print("Total elements in the train set after filtering: ", len(df_train_q3) + len(df_train_q4) + len(df_train_q8) + len(df_train_q10) + len(df_train_q12) + len(df_train_q13) + len(df_train_q17) + len(df_train_q19))

df_final_train = pd.concat([df_train_q3, df_train_q4 , df_train_q8 , df_train_q10 , df_train_q12 , df_train_q13 , df_train_q17 , df_train_q19], ignore_index=True)
print("No of rows in df_final ", len(df_final_train))

#print(df_final)
list_of_questions_filtered_in_train = df_final_train['NL_Question'].tolist()
print("list_of_questions_filtered_in_train:", len(list_of_questions_filtered_in_train))










