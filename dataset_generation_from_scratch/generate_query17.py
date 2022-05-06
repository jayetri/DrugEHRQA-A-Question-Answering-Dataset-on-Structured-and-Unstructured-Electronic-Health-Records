import pandas as pd
import findspark
findspark.init('/home/jayetri/spark/spark2/spark-3.0.2-bin-hadoop3.2')
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SparkSession
from pyspark import SQLContext, SparkConf
sc = SparkContext('local')
spark = SparkSession(sc)

################################# Extracting data from parquet files ###########################################3
i2b2_lookup_extract = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/i2b2_lookup_extract/*.parquet")
i2b2_lookup_extract.createOrReplaceTempView("i2b2_lookup_extract")
i2b2_lookup = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/i2b2_lookup/*.parquet")
i2b2_lookup.createOrReplaceTempView("i2b2_lookup")

####################### Extracting data from parquet files of three mimic iii tables #########################################

prescriptions = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/prescriptions/*.parquet")
prescriptions.createOrReplaceTempView("prescriptions")
d_icd_diagnoses = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/d_icd_diagnoses/*.parquet")
d_icd_diagnoses.createOrReplaceTempView("d_icd_diagnoses")
diagnoses_icd = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/diagnoses_icd/*.parquet")
diagnoses_icd.createOrReplaceTempView("diagnoses_icd")




#UNSTRUCTURED DATA OUTPUT

result_unstr = spark.sql(""" select distinct hadm_id,upper(arg2_value) as drug,arg1_value as problem from i2b2_lookup_extract  where upper(type)='REASON-DRUG'   
""")


def execute_unstructured_query17():	
    unstr_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_unstr.rdd.collect():
        if((str(row[0])+"|"+row[1]) not in newdict):
            newdict[str(row[0])+"|"+row[1]] = row[2] 
        else:
            newdict[str(row[0])+"|"+row[1]] = str(newdict[str(row[0])+"|"+row[1]])+","+row[2]
        print('\r',row,end='')
    for nextValue in newdict:
    
        naturalQuestion = 'Why is the patient with admission id ' + nextValue[0:6] + ' been given ' + nextValue[7:len(nextValue)]
        query = 'SELECT L3.SHORT_TITLE FROM D_ICD_DIAGNOSES AS L3 WHERE L3.ICD9_CODE IN (SELECT L1.ICD9_CODE FROM DIAGNOSES_ICD AS L1 INNER JOIN PRESCRIPTIONS AS L2 ON L1.HADM_ID = L2.HADM_ID WHERE L1.HADM_ID = '+nextValue[0:6]+' AND L2.DRUG = "'+nextValue[7:len(nextValue)]+'")'

        answer = "(" + newdict[nextValue] +")"
    
        unstr_df = unstr_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return unstr_df
        
unstr_df = execute_unstructured_query17()
unstr_df['NL_Question'] = unstr_df['NL_Question'].str.upper()
unstr_df['SQL_Query'] = unstr_df['SQL_Query'].str.upper()
unstr_df['Answer'] = unstr_df['Answer'].str.upper()




#STRUCTURED DATA OUTPUT

result_str = spark.sql("""select l1.hadm_id,l2.drug,l3.short_title
from diagnoses_icd l1
inner join 
(select hadm_id,upper(drug) as drug from prescriptions intersect select hadm_id,upper(value) as drug from i2b2_lookup)l2
on l1.hadm_id=l2.hadm_id
inner join
d_icd_diagnoses l3
on l1.icd9_code = l3.icd9_code
""")


def execute_structured_query17():	
    str_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_str.rdd.collect():
        if((row[0]+"|"+row[1]) not in newdict):
            newdict[row[0]+"|"+row[1]] = row[2] 
        else:
            newdict[row[0]+"|"+row[1]] = str(newdict[row[0]+"|"+row[1]])+","+row[2]
        print('\r',row,end='')
    for nextValue in newdict:
    
        naturalQuestion = 'Why is the patient with admission id ' + nextValue[0:6] + ' been given ' + nextValue[7:len(nextValue)]

        query = 'SELECT L3.SHORT_TITLE FROM D_ICD_DIAGNOSES AS L3 WHERE L3.ICD9_CODE IN (SELECT L1.ICD9_CODE FROM DIAGNOSES_ICD AS L1 INNER JOIN PRESCRIPTIONS AS L2 ON L1.HADM_ID = L2.HADM_ID WHERE L1.HADM_ID = '+nextValue[0:6]+' AND L2.DRUG = "'+nextValue[7:len(nextValue)]+'")'

        answer = "(" + newdict[nextValue] +")"
    
        str_df = str_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return str_df
        
str_df = execute_structured_query17()
str_df['NL_Question'] = str_df['NL_Question'].str.upper()
str_df['SQL_Query'] = str_df['SQL_Query'].str.upper()
str_df['Answer'] = str_df['Answer'].str.upper()



#COMBINE BOTH RESULTS

result_df = pd.merge(unstr_df, str_df, how='outer', on=['NL_Question', 'SQL_Query'])
result_df.columns = ["NL_Question","SQL_Query","Answer_Unstructured","Answer_Structured"]

df = result_df[["NL_Question","SQL_Query","Answer_Structured","Answer_Unstructured"]]

print(df)
df.to_excel("non_paraphrased_queries/query17.xlsx", index = False,  header=True)
