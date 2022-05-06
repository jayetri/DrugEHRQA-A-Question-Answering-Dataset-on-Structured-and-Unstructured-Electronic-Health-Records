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

result_unstr = spark.sql(""" select l1.hadm_id,l1.reason,l1.drug,l2.dosage from
(select distinct hadm_id,upper(arg1_value) as reason,upper(arg2_value) as drug from i2b2_lookup_extract  where upper(type)='REASON-DRUG' )l1
inner join
(select distinct hadm_id,arg1_value as dosage,upper(arg2_value) as drug from i2b2_lookup_extract  where upper(type)='DOSAGE-DRUG' )l2
on l1.hadm_id=l2.hadm_id and l1.drug=l2.drug
""")


def execute_unstructured_query21():	
    unstr_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_unstr.rdd.collect():
        if(row[3] ):
            if((str(row[0])+"|"+row[1]) not in newdict):
                newdict[str(row[0])+"|"+row[1]] = row[2] + ":" +row[3]  
            else:
                newdict[str(row[0])+"|"+row[1]] = str(newdict[str(row[0])+"|"+row[1]])+", "+row[2] + ":" +row[3] 
        print('\r',row,end='')

    for nextValue in newdict:
    
        naturalQuestion = 'List all the medicines and their dosages prescribed to the patient with admission id = ' + nextValue[0:6] + ' for ' + nextValue[7:len(nextValue)]

        query ='SELECT Y.DRUG, Y.DOSE_VAL_RX, Y.DOSE_UNIT_RX FROM PRESCRIPTIONS AS Y WHERE Y.HADM_ID = (SELECT L1.HADM_ID FROM DIAGNOSES_ICD AS L1 INNER JOIN D_ICD_DIAGNOSES AS L2 ON L1.ICD9_CODE = L2.ICD9_CODE WHERE L1.HADM_ID = ' + nextValue[0:6] + ' AND L2.SHORT_TITLE = "'+nextValue[7:len(nextValue)]+'")'
        
        answer = "(" + newdict[nextValue] +")"
    
        unstr_df = unstr_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return unstr_df
        
unstr_df = execute_unstructured_query21()
unstr_df['NL_Question'] = unstr_df['NL_Question'].str.upper()
unstr_df['SQL_Query'] = unstr_df['SQL_Query'].str.upper()
unstr_df['Answer'] = unstr_df['Answer'].str.upper()


#STRUCTURED DATA OUTPUT

result_str = spark.sql("""select l1.hadm_id,l3.short_title,l2.drug,l2.dose_val_rx,l2.dose_unit_rx
from diagnoses_icd l1
inner join 
(select hadm_id,upper(drug) as drug,dose_val_rx,dose_unit_rx from prescriptions where concat(hadm_id,upper(drug)) in ( select distinct concat(hadm_id,nvl(upper(value),"")) as drug from i2b2_lookup) )l2
on l1.hadm_id=l2.hadm_id
inner join
d_icd_diagnoses l3
on l1.icd9_code = l3.icd9_code
""")


def execute_structured_query21():	
    str_df = pd.DataFrame({'NL_Question': [], 'SQL_Query': [], 'Answer': []})
    newdict={}
    for row in result_str.rdd.collect():
        if(row[3] and row[4]):
            if((row[0]+"|"+row[1]) not in newdict):
                newdict[row[0]+"|"+row[1]] = row[2] + ":" +row[3] +" "+row[4] 
            else:
                newdict[row[0]+"|"+row[1]] = str(newdict[row[0]+"|"+row[1]])+", "+row[2] + ":" +row[3] +" "+row[4] 
        print('\r',row,end='')

    for nextValue in newdict:
    
        naturalQuestion = 'List all the medicines and their dosages prescribed to the patient with admission id = ' + nextValue[0:6] + ' for ' + nextValue[7:len(nextValue)]

        query ='SELECT Y.DRUG, Y.DOSE_VAL_RX, Y.DOSE_UNIT_RX FROM PRESCRIPTIONS AS Y WHERE Y.HADM_ID = (SELECT L1.HADM_ID FROM DIAGNOSES_ICD AS L1 INNER JOIN D_ICD_DIAGNOSES AS L2 ON L1.ICD9_CODE = L2.ICD9_CODE WHERE L1.HADM_ID = ' + nextValue[0:6] + ' AND L2.SHORT_TITLE = "'+nextValue[7:len(nextValue)]+'")'

        answer = "(" + newdict[nextValue] +")"
    
        str_df = str_df.append({'NL_Question': naturalQuestion, 'SQL_Query': query, 'Answer': answer}, ignore_index=True)
    return str_df
        
str_df = execute_structured_query21()
str_df['NL_Question'] = str_df['NL_Question'].str.upper()
str_df['SQL_Query'] = str_df['SQL_Query'].str.upper()
str_df['Answer'] = str_df['Answer'].str.upper()



#COMBINE BOTH RESULTS

result_df = pd.merge(unstr_df, str_df, how='outer', on=['NL_Question', 'SQL_Query'])
result_df.columns = ["NL_Question","SQL_Query","Answer_Unstructured","Answer_Structured"]

df = result_df[["NL_Question","SQL_Query","Answer_Structured","Answer_Unstructured"]]
print(df)
df.to_excel("non_paraphrased_queries/query21.xlsx", index = False,  header=True)


