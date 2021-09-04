# DrugEHRQA-A-Question-Answering-Dataset-on-Structured-and-Unstructured-Electronic-Health-Records
DrugEHRQA is the first question answering (QA) dataset containing question-answers from both structured tables and discharge summaries of MIMIC-III. It contains medicine-related queries on patient records.

The DrugEHRQA dataset can be accessed as follows:

a.Our QA dataset on the structured tables of MIMIC-III can be accessed through Physionet (https://physionet.org/). Download it    in the "structured_queries" directory. 

b.The QA pairs retrieved from the unstructured notes of MIMIC-III can be downloaded from the n2c2 repository (https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). Save it in the "unstructured_queries" directory.
 Both these datasets are combined to generate a multimodal QA dataset (DrugEHRQA). We had to submit it in different platforms for license issues. 

c.Run the script join_structured_and_unstructured.py to join both the datasets dowloaded from the two sources.

d.In order to generate the selected multimodal answer, run the script generate_multimodal_dataset.py using the following commands: 

    python generate_multimodal_dataset.py -i merged_queries/query1.xlsx -o outputs/query1.xlsx -a 0 -p 0
    python generate_multimodal_dataset.py --input_file=merged_queries/query2.xlsx --output_file=outputs/query2.xlsx --annotation_flag=1 --column_priority=0
    python generate_multimodal_dataset.py -i merged_queries/query3.xlsx -o outputs/query3.xlsx -a 2 -p 0
    python generate_multimodal_dataset.py -i merged_queries/query4.xlsx -o outputs/query4.xlsx -a 2 -p 0
    python generate_multimodal_dataset.py -i merged_queries/query5.xlsx -o outputs/query5.xlsx -a 1 -p 0
    python generate_multimodal_dataset.py -i merged_queries/query6.xlsx -o outputs/query6.xlsx -a 1 -p 1
    python generate_multimodal_dataset.py -i merged_queries/query7.xlsx -o outputs/query7.xlsx -a 1 -p 1
    python generate_multimodal_dataset.py -i merged_queries/query8.xlsx -o outputs/query8.xlsx -a 1 -p 1
    python generate_multimodal_dataset.py -i merged_queries/query9.xlsx -o outputs/query9.xlsx -a 3 -p 1

Note: This dataset is submitted as part of 'NeurIPS 2021 Track Datasets and Benchmarks Round 2'. The code for dataset generation from scratch will be updated here, after it is accepted by NeurIPS. 