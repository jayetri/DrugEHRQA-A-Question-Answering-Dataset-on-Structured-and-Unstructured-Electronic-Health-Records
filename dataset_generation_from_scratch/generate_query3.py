import pandas as pd
import findspark
findspark.init('/home/jayetri/spark/spark2/spark-3.0.2-bin-hadoop3.2')
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SparkSession
from pyspark import SQLContext, SparkConf
sc = SparkContext('local')
spark = SparkSession(sc)

'''
####################### Generating parquet files #########################################
df = spark.read.format('csv').options(header= 'true').load('i2b2_data_extracted.csv')
df.show()
df.repartition(1).write.mode('overwrite').parquet('/home/jayetri/redo_drug_dataset/spark-warehouse/i2b2_lookup_extract')


df_i2b2 = spark.read.format('csv').options(header= 'true').load('i2b2_data.csv')
df_i2b2.show()
df_i2b2.repartition(1).write.mode('overwrite').parquet('/home/jayetri/redo_drug_dataset/spark-warehouse/i2b2_lookup')


####################### Generating parquet files from 3 mimic iii tables #########################################
prescription_df = spark.read.format('csv').options(header= 'true').load('/home/jayetri/complete_mimiciii_dataset/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv')
prescription_df.show()
prescription_df.repartition(1).write.mode('overwrite').parquet('/home/jayetri/redo_drug_dataset/spark-warehouse/prescriptions')
'''

################################# Extracting data from parquet files ###########################################3
i2b2_lookup_extract = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/i2b2_lookup_extract/*.parquet")
i2b2_lookup_extract.createOrReplaceTempView("i2b2_lookup_extract")
i2b2_lookup = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/i2b2_lookup/*.parquet")
i2b2_lookup.createOrReplaceTempView("i2b2_lookup")

####################### Extracting data from parquet files of three mimic iii tables #########################################

prescriptions = spark.read.load("/home/jayetri/redo_drug_dataset/spark-warehouse/prescriptions/*.parquet")
prescriptions.createOrReplaceTempView("prescriptions")




#UNSTRUCTURED DATA OUTPUT

result_unstr = spark.sql(""" select distinct hadm_id as hadm_id,upper(value) as value from i2b2_lookup_extract where upper(type)='DRUG'        
""")


def execute_unstructured_query3():	
    unstr_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_unstr.rdd.collect():
    
        if(row[0] not in newdict):
            newdict[row[0]] = row[1] 
        else:
            newdict[row[0]] = newdict[row[0]]+","+row[1]
        
    for nextValue in newdict:
    
        naturalQuestion = "What are the list of medicines prescribed to the patient with admission id " + str(nextValue) 

        query ='SELECT PRESCRIPTIONS.DRUG FROM PRESCRIPTIONS WHERE PRESCRIPTIONS.HADM_ID = '+str(nextValue)

        answer = "(" + newdict[nextValue] +  ")"
    
        unstr_df = unstr_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)

    return unstr_df
        
unstr_df = execute_unstructured_query3()
unstr_df['NL_Question'] = unstr_df['NL_Question'].str.upper()
unstr_df['SQL_Query'] = unstr_df['SQL_Query'].str.upper()
unstr_df['Answer'] = unstr_df['Answer'].str.upper()



#STRUCTURED DATA OUTPUT

result_str = spark.sql("""
        select distinct hadm_id as hadm_id,upper(value) as value from i2b2_lookup 
            intersect 
        select hadm_id,upper(drug) as drug from prescriptions 
""")


def execute_structured_query3():	
    str_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_str.rdd.collect():
    
        if(row[0] not in newdict):
            newdict[row[0]] = row[1] 
        else:
            newdict[row[0]] = newdict[row[0]]+","+row[1]
        
    for nextValue in newdict:
    
        naturalQuestion = "What are the list of medicines prescribed to the patient with admission id " + str(nextValue) 

        query ='SELECT PRESCRIPTIONS.DRUG FROM PRESCRIPTIONS WHERE PRESCRIPTIONS.HADM_ID = '+str(nextValue)

        answer = "(" + newdict[nextValue] +  ")"
    
        str_df = str_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return str_df
        
str_df = execute_structured_query3()
str_df['NL_Question'] = str_df['NL_Question'].str.upper()
str_df['SQL_Query'] = str_df['SQL_Query'].str.upper()
str_df['Answer'] = str_df['Answer'].str.upper()





#COMBINE BOTH RESULTS

result_df = pd.merge(unstr_df, str_df, how='outer', on=['NL_Question', 'SQL_Query'])
result_df.columns = ["NL_Question","SQL_Query","Answer_Unstructured","Answer_Structured"]

df = result_df[["NL_Question","SQL_Query","Answer_Structured","Answer_Unstructured"]]
print(df)
df.to_excel("non_paraphrased_queries/query3.xlsx", index = False,  header=True)

