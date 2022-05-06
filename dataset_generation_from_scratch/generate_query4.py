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

result_unstr = spark.sql(""" select distinct hadm_id,upper(arg2_value) as drug,arg1_value from i2b2_lookup_extract  where upper(type)='STRENGTH-DRUG'     
""")


def execute_unstructured_query4():	
    unstr_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_unstr.rdd.collect():
        if((str(row[0])+"|"+row[1]) not in newdict):
            newdict[str(row[0])+"|"+row[1]] = row[2] 
        else:
            newdict[str(row[0])+"|"+row[1]] = str(newdict[str(row[0])+"|"+row[1]])+","+row[2]
    
    for nextValue in newdict:
    
        naturalQuestion = 'What is the drug strength of ' + nextValue[7:len(nextValue)] + ' prescribed to the patient with admission id ' + nextValue[0:6]

        query ='select prescriptions.prod_strength from prescriptions where prescriptions.hadm_id = '+nextValue[0:6]+' and prescriptions.drug = "'+nextValue[7:len(nextValue)] + '"'

        answer = "(" + newdict[nextValue] +  ")"
    
        unstr_df = unstr_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return unstr_df
        
unstr_df = execute_unstructured_query4()
unstr_df['NL_Question'] = unstr_df['NL_Question'].str.upper()
unstr_df['SQL_Query'] = unstr_df['SQL_Query'].str.upper()
unstr_df['Answer'] = unstr_df['Answer'].str.upper()





#STRUCTURED DATA OUTPUT

result_str = spark.sql("""
        select distinct ps.hadm_id as hadm_id,upper(ps.drug) as drug,ps.PROD_STRENGTH from prescriptions ps inner join i2b2_lookup i2b2 on ps.hadm_id=i2b2.hadm_id and upper(ps.drug) = upper(i2b2.value)
            
""")


def execute_structured_query4():	
    str_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_str.rdd.collect():
        print('\r',row,end='')
        if(row[2]):
            if((row[0]+"|"+row[1]) not in newdict):
                newdict[row[0]+"|"+row[1]] = row[2] 
            else:
                newdict[row[0]+"|"+row[1]] = str(newdict[row[0]+"|"+row[1]])+","+row[2]
    for nextValue in newdict:
    
        naturalQuestion = 'What is the drug strength of ' + nextValue[7:len(nextValue)] + ' prescribed to the patient with admission id ' + nextValue[0:6]

        query ='select prescriptions.prod_strength from prescriptions where prescriptions.hadm_id = '+nextValue[0:6]+' and prescriptions.drug = "'+nextValue[7:len(nextValue)] + '"'

        answer = "(" + newdict[nextValue] +  ")"
    
        str_df = str_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return str_df
        
str_df = execute_structured_query4()
str_df['NL_Question'] = str_df['NL_Question'].str.upper()
str_df['SQL_Query'] = str_df['SQL_Query'].str.upper()
str_df['Answer'] = str_df['Answer'].str.upper()




#COMBINE BOTH RESULTS

result_df = pd.merge(unstr_df, str_df, how='outer', on=['NL_Question', 'SQL_Query'])
result_df.columns = ["NL_Question","SQL_Query","Answer_Unstructured","Answer_Structured"]

df = result_df[["NL_Question","SQL_Query","Answer_Structured","Answer_Unstructured"]]

print(df)
df.to_excel("non_paraphrased_queries/query4.xlsx", index = False,  header=True)

