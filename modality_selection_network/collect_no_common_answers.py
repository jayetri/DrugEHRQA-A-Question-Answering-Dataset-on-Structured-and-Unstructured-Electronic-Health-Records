import pandas as pd
import numpy as np
import json
import re

# Function takes list of strings as input.
# Returns list of preprocessed strings (removes spaces, '(S)', and 'S') and a mapping to original string.
def process_strings(string_list):
    true_value_mapping = {}
    return_list = list()
    for i in range(len(string_list)):
        if string_list[i] == "":
            continue
        true_value = string_list[i]
        return_list.append(string_list[i].replace(" ", ""))
        if return_list[len(return_list) - 1][-3: len(return_list[len(return_list) - 1])] == "(S)":
            return_list[len(return_list) - 1] = return_list[len(return_list) - 1][0: len(return_list[len(return_list) -
                                                                                                     1]) - 3]
        elif return_list[len(return_list) - 1][-1] == 'S':
            return_list[len(return_list) - 1] = return_list[len(return_list) - 1][0: len(return_list[len(return_list) -
                                                                                                     1]) - 1]
        true_value_mapping.update({return_list[len(return_list) - 1]: true_value})
    return return_list, true_value_mapping


# Function takes cell contents of structured or unstructured annotation from query file as input.
# Returns list of preprocessed strings (removes spaces, '(S)', and 'S') and a mapping to original string.
def get_values(cell):
    if cell[0] == '(':
        cell = cell[1: len(cell)]
    if cell[-3:len(cell)] != "(S)" and cell[len(cell)-1] == ')':
        cell = cell[0: len(cell)-1]
    string_list = cell.split(',')
    string_list,  true_value_mapping = process_strings(string_list)
    return string_list, true_value_mapping

# Function returns the common strings (strict equality) from 2 string lists
def get_common_values(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if set1 & set2:
        return list(set1 & set2)
    return list()



###########################Query 3 start #####################################################
print("Start Query3----------------------------------------------------------------------------------->")
df_q3 = pd.read_excel ('added_label/query3.xlsx')
total = 0
intersect_count_3 = 0
no_intersect_count_3 = 0
print("No of queries in q3: ", len(df_q3))  
for index, row in df_q3.iterrows():                
    structured_raw = df_q3.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q3.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured: ", structured)
    #print("unstructured: ", unstructured)
    common_values = get_common_values(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q3.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_3 = no_intersect_count_3 + 1
    else:
        df_q3.at[index, 'intersecting_answers'] = '1'
        intersect_count_3 = intersect_count_3 + 1

print("In query 3, out of total ", str(total), " queries, there are ", str(intersect_count_3), " intersecting queries and ", str(no_intersect_count_3), " non-intersecting queries.")


df_intersecting_q3 = df_q3[df_q3['intersecting_answers'] == '1'] 
df_intersecting_q3.to_excel("intersecting_answers/query3.xlsx", index = False,  header=True)

df_non_intersecting_q3 = df_q3[df_q3['intersecting_answers'] == '0'] 
df_non_intersecting_q3.to_excel("non_intersecting_answers/query3.xlsx", index = False,  header=True)

df_train_3 = df_intersecting_q3
df_train_3 = df_train_3.sample(frac=1)
df_test_3 = df_non_intersecting_q3
print( "Query3, test= ", str(len(df_test_3)), " and train = ", str(len(df_train_3)) )
df_train_3.to_excel("non_intersecting_answers/train_and_dev_set/query3.xlsx", index = False,  header=True)
df_test_3.to_excel("non_intersecting_answers/test_set_not_filtered/query3.xlsx", index = False,  header=True)

###########################Query 4 start #####################################################

# Returns True if only one number each exists in the strings and they are equal. False for all other cases
def get_equality_numeric(str1, str2):
    n_str1 = ""
    i = 0
    while i < len(str1):
        if '0' <= str1[i] <= '9':
            if n_str1 != "":
                return False
            n_str1 += str1[i]
            i += 1
            while i < len(str1) and '0' <= str1[i] <= '9':
                n_str1 += str1[i]
                i += 1
        elif i >= len(str1):
            break
        else:
            i += 1

    n_str2 = ""
    i = 0
    while i < len(str2):
        if '0' <= str2[i] <= '9':
            if n_str2 != "":
                return False
            n_str2 += str2[i]
            i += 1
            while i < len(str2) and '0' <= str2[i] <= '9':
                n_str2 += str2[i]
                i += 1
        elif i >= len(str2):
            break
        else:
            i += 1
    if n_str1 == "" or n_str2 == "":
        return False
    if n_str1 == n_str2:
        return True
    return False


# Function returns strings which satisfy initial substring match and number embedded in string.
# Prioritises representation given in list1
def get_common_values_with_initial_substring_match(list1, list2):
    set1 = set(list1)
    res = list()
    while len(set1) > 0:  # loop checks if shorter sting is contained in the larger string. And if numeric equality exists
        str1 = set1.pop()
        for j in list2:
            if get_equality_numeric(str1, j):
                res.append(str1)
            elif len(str1) >= len(j):
                if j == str1[0:len(j)]:
                    res.append(str1)
            else:
                if str1 == j[0:len(str1)]:
                    res.append(str1)
    return res

print("Start Query4----------------------------------------------------------------------------------->")
df_q4 = pd.read_excel ('added_label/query4.xlsx')
total = 0
intersect_count_4 = 0
no_intersect_count_4 = 0
print("No of queries in q4: ", len(df_q4))  
for index, row in df_q4.iterrows():                
    structured_raw = df_q4.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q4.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured_raw: ", structured_raw)
    #print("unstructured_raw: ", unstructured_raw)
    common_values = get_common_values_with_initial_substring_match(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q4.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_4 = no_intersect_count_4 + 1

    else:
        df_q4.at[index, 'intersecting_answers'] = '1'
        intersect_count_4 = intersect_count_4 + 1

print("In query 4, out of total ", str(total), " queries, there are ", str(intersect_count_4), " intersecting queries and ", str(no_intersect_count_4), " non-intersecting queries.")
df_intersecting_q4 = df_q4[df_q4['intersecting_answers'] == '1'] 
df_intersecting_q4.to_excel("intersecting_answers/query4.xlsx", index = False,  header=True)

df_non_intersecting_q4 = df_q4[df_q4['intersecting_answers'] == '0'] 
df_non_intersecting_q4.to_excel("non_intersecting_answers/query4.xlsx", index = False,  header=True)

df_test_4 = df_non_intersecting_q4.sample(n=295)
df_test_4 = df_test_4.sample(frac=1)

df_leftover_4 = df_non_intersecting_q4[~df_non_intersecting_q4.isin(df_test_4)].dropna()
df_train_4 = pd.concat([df_leftover_4, df_intersecting_q4], axis=0)
df_train_4 = df_train_4.sample(frac=1)

print( "Query4, test= ", str(len(df_test_4)), " and train = ", str(len(df_train_4)) )
df_train_4.to_excel("non_intersecting_answers/train_and_dev_set/query4.xlsx", index = False,  header=True)
df_test_4.to_excel("non_intersecting_answers/test_set_not_filtered/query4.xlsx", index = False,  header=True)

###########################Query 8 start #####################################################

print("Start Query8----------------------------------------------------------------------------------->")

# Function works same as get_common_values_with_initial_substring_match() with added comparison for acronyms
def get_common_values_with_initial_substring_and_acronym_match(list1, list2):
    set1 = set(list1)
    res = list()
    while len(set1) > 0:
        str1 = set1.pop()
        if str1 == "IV".lower() or str1 == "INTRAVENOU".lower():
            if "IV".lower() in list2 or "INTRAVENOU".lower() in list2:
                res.append(str1)
                continue
        if str1 == "IH".lower() or str1 == "INHALATION".lower():
            if "IH".lower() in list2 or "INHALATION".lower() in list2:
                res.append(str1)
                continue
        for j in list2:
            if get_equality_numeric(str1, j):
                res.append(str1)
            elif len(str1) >= len(j):
                if j == str1[0:len(j)]:
                    res.append(str1)
            else:
                if str1 == j[0:len(str1)]:
                    res.append(str1)
    return res

df_q8 = pd.read_excel ('added_label/query8.xlsx')
total = 0
intersect_count_8 = 0
no_intersect_count_8 = 0
print("No of queries in q8: ", len(df_q8))  
for index, row in df_q8.iterrows():                
    structured_raw = df_q8.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q8.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured_raw: ", structured_raw)
    #print("unstructured_raw: ", unstructured_raw)
    common_values = get_common_values_with_initial_substring_and_acronym_match(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q8.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_8 = no_intersect_count_8 + 1

    else:
        df_q8.at[index, 'intersecting_answers'] = '1'
        intersect_count_8 = intersect_count_8 + 1

print("In query 8, out of total ", str(total), " queries, there are ", str(intersect_count_8), " intersecting queries and ", str(no_intersect_count_8), " non-intersecting queries.")
df_intersecting_q8 = df_q8[df_q8['intersecting_answers'] == '1'] 
df_intersecting_q8.to_excel("intersecting_answers/query8.xlsx", index = False,  header=True)

df_non_intersecting_q8 = df_q8[df_q8['intersecting_answers'] == '0'] 
df_non_intersecting_q8.to_excel("non_intersecting_answers/query8.xlsx", index = False,  header=True)


df_test_8 = df_non_intersecting_q8.sample(n=222)
df_leftover_8 = df_non_intersecting_q8[~df_non_intersecting_q8.isin(df_test_8)].dropna()
df_train_8 = pd.concat([df_leftover_8, df_intersecting_q8], axis=0)
df_train_8 = df_train_8.sample(frac=1)
df_test_8 = df_test_8.sample(frac=1)

print( "Query8, test= ", str(len(df_test_8)), " and train = ", str(len(df_train_8)) )
df_train_8.to_excel("non_intersecting_answers/train_and_dev_set/query8.xlsx", index = False,  header=True)
df_test_8.to_excel("non_intersecting_answers/test_set_not_filtered/query8.xlsx", index = False,  header=True)


###########################Query 10 start #####################################################

print("Start Query10----------------------------------------------------------------------------------->")


df_q10 = pd.read_excel ('added_label/query10.xlsx')
total = 0
intersect_count_10 = 0
no_intersect_count_10 = 0
print("No of queries in q10: ", len(df_q10))  
for index, row in df_q10.iterrows():                
    structured_raw = df_q10.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q10.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured: ", structured)
    #print("unstructured: ", unstructured)
    common_values = get_common_values_with_initial_substring_and_acronym_match(structured, unstructured)  # get common values between structured and unstructured
    #print("common values: ", common_values)
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q10.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_10 = no_intersect_count_10 + 1

    else:
        #print("MATCHED")
        df_q10.at[index, 'intersecting_answers'] = '1'
        intersect_count_10 = intersect_count_10 + 1

print("In query 10, out of total ", str(total), " queries, there are ", str(intersect_count_10), " intersecting queries and ", str(no_intersect_count_10), " non-intersecting queries.")
df_intersecting_q10 = df_q10[df_q10['intersecting_answers'] == '1'] 
df_intersecting_q10.to_excel("intersecting_answers/query10.xlsx", index = False,  header=True)

df_non_intersecting_q10 = df_q10[df_q10['intersecting_answers'] == '0'] 
df_non_intersecting_q10.to_excel("non_intersecting_answers/query10.xlsx", index = False,  header=True)

df_test_10 = df_non_intersecting_q10.sample(n=281)
df_leftover_10 = df_non_intersecting_q10[~df_non_intersecting_q10.isin(df_test_10)].dropna()
df_train_10 = pd.concat([df_leftover_10, df_intersecting_q10], axis=0)
df_train_10 = df_train_10.sample(frac=1)
df_test_10 = df_test_10.sample(frac=1)
print( "Query10, test= ", str(len(df_test_10)), " and train = ", str(len(df_train_10)) )
df_train_10.to_excel("non_intersecting_answers/train_and_dev_set/query10.xlsx", index = False,  header=True)
df_test_10.to_excel("non_intersecting_answers/test_set_not_filtered/query10.xlsx", index = False,  header=True)


###########################Query 12 start #####################################################

print("Start Query12----------------------------------------------------------------------------------->")

df_q12 = pd.read_excel ('added_label/query12.xlsx')
total = 0
intersect_count_12 = 0
no_intersect_count_12 = 0
print("No of queries in q12: ", len(df_q12))  
for index, row in df_q12.iterrows():                
    structured_raw = df_q12.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q12.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured_raw: ", structured_raw)
    #print("unstructured_raw: ", unstructured_raw)
    common_values = get_common_values_with_initial_substring_match(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q12.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_12 = no_intersect_count_12 + 1

    else:
        df_q12.at[index, 'intersecting_answers'] = '1'
        intersect_count_12 = intersect_count_12 + 1

print("In query 12, out of total ", str(total), " queries, there are ", str(intersect_count_12), " intersecting queries and ", str(no_intersect_count_12), " non-intersecting queries.")
df_intersecting_q12 = df_q12[df_q12['intersecting_answers'] == '1'] 
df_intersecting_q12.to_excel("intersecting_answers/query12.xlsx", index = False,  header=True)

df_non_intersecting_q12 = df_q12[df_q12['intersecting_answers'] == '0'] 
df_non_intersecting_q12.to_excel("non_intersecting_answers/query12.xlsx", index = False,  header=True)


df_test_12 = df_non_intersecting_q12.sample(n=232)
df_leftover_12 = df_non_intersecting_q12[~df_non_intersecting_q12.isin(df_test_12)].dropna()
df_train_12 = pd.concat([df_leftover_12, df_intersecting_q12], axis=0)
df_train_12 = df_train_12.sample(frac=1)
df_test_12 = df_test_12.sample(frac=1)
print( "Query12, test= ", str(len(df_test_12)), " and train = ", str(len(df_train_12)) )

df_train_12.to_excel("non_intersecting_answers/train_and_dev_set/query12.xlsx", index = False,  header=True)
df_test_12.to_excel("non_intersecting_answers/test_set_not_filtered/query12.xlsx", index = False,  header=True)

###########################Query 13 start #####################################################

print("Start Query13----------------------------------------------------------------------------------->")

df_q13 = pd.read_excel ('added_label/query13.xlsx')
total = 0
intersect_count_13 = 0
no_intersect_count_13 = 0
print("No of queries in q13: ", len(df_q13))  
for index, row in df_q13.iterrows():                
    structured_raw = df_q13.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q13.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured_raw: ", structured_raw)
    #print("unstructured_raw: ", unstructured_raw)
    common_values = get_common_values_with_initial_substring_match(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q13.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_13 = no_intersect_count_13 + 1

    else:
        df_q13.at[index, 'intersecting_answers'] = '1'
        intersect_count_13 = intersect_count_13 + 1

print("In query 13, out of total ", str(total), " queries, there are ", str(intersect_count_13), " intersecting queries and ", str(no_intersect_count_13), " non-intersecting queries.")
df_intersecting_q13 = df_q13[df_q13['intersecting_answers'] == '1'] 
df_intersecting_q13.to_excel("intersecting_answers/query13.xlsx", index = False,  header=True)

df_non_intersecting_q13 = df_q13[df_q13['intersecting_answers'] == '0'] 
df_non_intersecting_q13.to_excel("non_intersecting_answers/query13.xlsx", index = False,  header=True)


df_test_13 = df_non_intersecting_q13.sample(n=28)
df_leftover_13 = df_non_intersecting_q13[~df_non_intersecting_q13.isin(df_test_13)].dropna()
df_train_13 = pd.concat([df_leftover_13, df_intersecting_q13], axis=0)
df_train_13 = df_train_13.sample(frac=1)
df_test_13 = df_test_13.sample(frac=1)
print( "Query13, test= ", str(len(df_test_13)), " and train = ", str(len(df_train_13)) )
df_train_13.to_excel("non_intersecting_answers/train_and_dev_set/query13.xlsx", index = False,  header=True)
df_test_13.to_excel("non_intersecting_answers/test_set_not_filtered/query13.xlsx", index = False,  header=True)


###########################Query 17 start #####################################################

print("Start Query17----------------------------------------------------------------------------------->")

df_q17 = pd.read_excel ('added_label/query17.xlsx')
total = 0
intersect_count_17 = 0
no_intersect_count_17 = 0
print("No of queries in q17: ", len(df_q17))  
for index, row in df_q17.iterrows():                
    structured_raw = df_q17.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q17.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured_raw: ", structured_raw)
    #print("unstructured_raw: ", unstructured_raw)
    common_values = get_common_values(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q17.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_17 = no_intersect_count_17 + 1

    else:
        df_q17.at[index, 'intersecting_answers'] = '1'
        intersect_count_17 = intersect_count_17 + 1

print("In query 17, out of total ", str(total), " queries, there are ", str(intersect_count_17), " intersecting queries and ", str(no_intersect_count_17), " non-intersecting queries.")
df_intersecting_q17 = df_q17[df_q17['intersecting_answers'] == '1'] 
df_intersecting_q17.to_excel("intersecting_answers/query17.xlsx", index = False,  header=True)

df_non_intersecting_q17 = df_q17[df_q17['intersecting_answers'] == '0'] 
df_non_intersecting_q17.to_excel("non_intersecting_answers/query17.xlsx", index = False,  header=True)


df_test_17 = df_non_intersecting_q17.sample(n=150)
df_leftover_17 = df_non_intersecting_q17[~df_non_intersecting_q17.isin(df_test_17)].dropna()
df_train_17 = pd.concat([df_leftover_17, df_intersecting_q17], axis=0)
df_train_17 = df_train_17.sample(frac=1)
df_test_17 = df_test_17.sample(frac=1)
print( "Query17, test= ", str(len(df_test_17)), " and train = ", str(len(df_train_17)) )

df_train_17.to_excel("non_intersecting_answers/train_and_dev_set/query17.xlsx", index = False,  header=True)
df_test_17.to_excel("non_intersecting_answers/test_set_not_filtered/query17.xlsx", index = False,  header=True)
###########################Query 19 start #####################################################

print("Start Query19----------------------------------------------------------------------------------->")

df_q19 = pd.read_excel ('added_label/query19.xlsx')
total = 0
intersect_count_19 = 0
no_intersect_count_19 = 0
print("No of queries in q19: ", len(df_q19))  
for index, row in df_q19.iterrows():                
    structured_raw = df_q19.at[index, 'Answer_Structured']
    structured1, structured_true_value_mapping = get_values(structured_raw)
    structured = [x.lower() for x in structured1]
    total += 1
    unstructured_raw = df_q19.at[index, 'Answer_Unstructured']
    unstructured1, unstructured_true_value_mapping = get_values(unstructured_raw)
    unstructured = [x.lower() for x in unstructured1]
    #print("structured_raw: ", structured_raw)
    #print("unstructured_raw: ", unstructured_raw)
    common_values = get_common_values(structured, unstructured)  # get common values between structured and unstructured
    if len(common_values) == 0:
        #print("NOT MATCHED")
        df_q19.at[index, 'intersecting_answers'] = '0'
        no_intersect_count_19 = no_intersect_count_19 + 1

    else:
        df_q19.at[index, 'intersecting_answers'] = '1'
        intersect_count_19 = intersect_count_19 + 1

print("In query 19, out of total ", str(total), " queries, there are ", str(intersect_count_19), " intersecting queries and ", str(no_intersect_count_19), " non-intersecting queries.")
df_intersecting_q19 = df_q19[df_q19['intersecting_answers'] == '1'] 
df_intersecting_q19.to_excel("intersecting_answers/query19.xlsx", index = False,  header=True)

df_non_intersecting_q19 = df_q19[df_q19['intersecting_answers'] == '0'] 
df_non_intersecting_q19.to_excel("non_intersecting_answers/query19.xlsx", index = False,  header=True)

df_test_19 = df_non_intersecting_q19.sample(n=12)
df_leftover_19 = df_non_intersecting_q19[~df_non_intersecting_q19.isin(df_test_19)].dropna()
df_train_19 = pd.concat([df_leftover_19, df_intersecting_q19], axis=0)
df_train_19 = df_train_19.sample(frac=1)
df_test_19 = df_test_19.sample(frac=1)
print( "Query19, test= ", str(len(df_test_19)), " and train = ", str(len(df_train_19)) )
df_train_19.to_excel("non_intersecting_answers/train_and_dev_set/query19.xlsx", index = False,  header=True)
df_test_19.to_excel("non_intersecting_answers/test_set_not_filtered/query19.xlsx", index = False,  header=True)


##########################Final computations-------------------------------------------------


print("Final computations------------------------------------------------------------")

total_intersecting_queries = len(df_intersecting_q3) + len(df_intersecting_q4) + len(df_intersecting_q8) + len(df_intersecting_q10) + len(df_intersecting_q12) + len(df_intersecting_q13) + len(df_intersecting_q17)+ len(df_intersecting_q19)
total_non_intersecting_queries = len(df_non_intersecting_q3) + len(df_non_intersecting_q4) + len(df_non_intersecting_q8) + len(df_non_intersecting_q10) + len(df_non_intersecting_q12) + len(df_non_intersecting_q13) + len(df_non_intersecting_q17)+ len(df_non_intersecting_q19)

print("Total number of questions with intersecting answers:", total_intersecting_queries)
print("Total number of questions with non_intersecting answers:", total_non_intersecting_queries)

print('/n')

train = pd.concat([df_train_3 , df_train_4 , df_train_8 , df_train_10 , df_train_12 , df_train_13 , df_train_17 , df_train_19], axis=0)
test = pd.concat([df_test_3 , df_test_4 , df_test_8 , df_test_10 , df_test_12 , df_test_13 , df_test_17 , df_test_19], axis=0)
train = train.sample(frac=1)
test = test.sample(frac=1)



print("Number of rows in train set: ", len(train))
print("Number of rows in test set: ", len(test))

train.to_excel("train.xlsx", index = False,  header=True)
test.to_excel("test.xlsx", index = False,  header=True)
train.to_csv("train.csv", index = False,  header=True)
test.to_csv("test.csv", index = False,  header=True)
