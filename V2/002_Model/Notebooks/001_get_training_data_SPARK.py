# %%
# !! {"metadata":{
# !!   "id":"cc-imports"
# !! }}

#<cc-imports>

import subprocess

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {
# !!       "byteLimit": 2048000,
# !!       "rowLimit": 10000
# !!     },
# !!     "inputWidgets": {},
# !!     "nuid": "44920190-9120-4ae8-858f-14f52b073315",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "6kA3x79eY801",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286574754,
# !!     "user_tz": 180,
# !!     "elapsed": 25186,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "7dd0267b-d367-4ca9-b554-0c6fa0f4bd5c"
# !! }}
import pickle
sub_p_res = subprocess.run(['pip', 'install', 'boto3'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
import boto3

sub_p_res = subprocess.run(['pip', 'install', 'fsspec'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
import fsspec

import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {
# !!       "byteLimit": 2048000,
# !!       "rowLimit": 10000
# !!     },
# !!     "inputWidgets": {},
# !!     "nuid": "2ba961a2-7e64-4844-8d78-c47e5f47c752",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "B17TKGR3Y803",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "164da65e-3d3a-4ba0-aa72-bdd2d5522a9e",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286590498,
# !!     "user_tz": 180,
# !!     "elapsed": 15748,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
sub_p_res = subprocess.run(['pip', 'install', 'pyspark'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
#sc = spark.sparkContext         # Since Spark 2.0 'spark' is a
#SparkSession object that is by default created upfront and available
#in Spark shell, you need to explicitly create SparkSession object by
#using builder
spark = SparkSession.builder.getOrCreate()
#sc = SparkContext().getOrCreate()
sc = SparkContext._active_spark_context #devuelve la instancia existente
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType, FloatType, ArrayType, DoubleType, StructType, StructField
sqlContext = SQLContext(sc,spark)

# %%
# !! {"metadata":{
# !!   "id": "wkefE1c1ZJjH",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286621192,
# !!     "user_tz": 180,
# !!     "elapsed": 30699,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "e127124b-c112-4777-8ec1-6cc7e73f03f7"
# !! }}
from google.colab import drive
drive.mount('/content/drive')

# %%
# !! {"metadata":{
# !!   "id": "VmKqcV0fh14s"
# !! }}
"""
**Cambiar path para correr desde otro lado**
"""

# %%
# !! {"metadata":{
# !!   "id": "h_yrzVlKbWdn",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286621192,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
import os
base_path = '/content/drive/MyDrive/openalex-institution-parsing/'

base_save_path = "/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/"

iteration_save_path = "/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/"


# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "47f9abef-de4b-45bd-8d74-3d30740105f2",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "445hVw9cY806"
# !! }}
"""
### Getting all data (From saved OpenAlex DB snapshot)
"""

# %%
# !! {"metadata":{
# !!   "id": "Lkhoxw7XiCWg"
# !! }}
"""

"""

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "bfbb7d3c-7adf-4a23-aaa8-39f6f9f38d7b",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "Mm-2Kv5sY808",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286631664,
# !!     "user_tz": 180,
# !!     "elapsed": 10476,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "f66eb9e6-39ad-4718-f0ba-fc9e07e7ee40"
# !! }}
#institutions = spark.read.parquet(f"{base_save_path}/OA_static_institutions_single_file.parquet") \
 #   .filter(F.col('ror_id')!='')

#institutions_df = pd.read_parquet("/content/drive/MyDrive/openalex-institution-parsing/Archivos/OA_static_institutions_single_file.parquet")

### ALTERNATIVA DE STATIC INSTITUTIONS: BAJAR CON PYALEX PARTE DEL OBJETO INSTITUTIONS
sub_p_res = subprocess.run(['pip', 'install', 'pyalex'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
import pyalex
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders

from pyalex import config

config.max_retries = 0
config.retry_backoff_factor = 0.1
config.retry_http_codes = [429, 500, 503]

pager = Institutions().filter(country_code="AR").paginate(method="page",per_page=200)

listaDeInstituciones = list()
for page in pager:
    print(len(page))
    listaDeInstituciones += page

print(len(listaDeInstituciones))

institutions_df=pd.DataFrame(listaDeInstituciones)
institutions_df.head(n=2)


institutions_df['affiliation_id'] = institutions_df['id'].apply(lambda x: x.split("/")[-1])
institutions_df['affiliation_id']=institutions_df['affiliation_id'].apply(lambda x: x.split("I")[-1])

institutions_df['ror_id'] = institutions_df['ror'].apply(lambda x: x.split("/")[-1])

##nstitutions_df['display_name'] = institutions_df['display_name_alternatives'] ##ojo lo comenté!
institutions_df.head(n=2)
print(institutions_df.shape)

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "06521a7d-58ae-4e31-a5ab-add6547b2145",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "2I1GRFM1Y809",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286631665,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
#from pyspark.sql import Row
# spark is from the previous example.
# Create a simple DataFrame, stored into a partition directory
#sc = spark.sparkContextinstitutions.cache().count()

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {
# !!       "byteLimit": 2048000,
# !!       "rowLimit": 10000
# !!     },
# !!     "inputWidgets": {},
# !!     "nuid": "98e62bb7-5d47-484e-9a8f-43a734fd5beb",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "dB_zAKKLY80_",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286631665,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
##STATIC AFFILIATION!
##utilizando el que estaba en drive
##affiliations = spark.read.parquet(f"{'/content/drive/MyDrive/openalex-institution-parsing/Archivos'}/static_affiliations.parquet")


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "XZUIe8gZosrC",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286822910,
# !!     "user_tz": 180,
# !!     "elapsed": 191251,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "850c84b8-fe09-40fc-b8f6-1e79aa530a95"
# !! }}
#### ALTERNATIVA DE STATIC AFFILIATION: BAJAR DESDE OPENALEX CON PYALEX
sub_p_res = subprocess.run(['pip', 'install', 'pyalex'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
import pyalex
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders

#pyalex.config.email = "mfartola.mincyt@gmail.com"
from pyalex import config
#config.max_retries = 0
#config.retry_backoff_factor = 0.1
#config.retry_http_codes = [429, 500, 503]

import pandas as pd
#argentina_works=Works().filter(authorships={"institutions": {"country_code": "AR"}}).get()
#argentina_works=pd.DataFrame(argentina_works)
#argentina_works['authorships'].head()
#argentina_works.columns
#argentina_works.head()

pager = Works().filter(authorships={"institutions": {"country_code": "AR"}}).paginate(method="page",per_page=200)

listaDeWorks = list()
for page in pager:
    print(len(page))
    listaDeWorks += page

#print(len(listaDeWorks))

argentina_works=pd.DataFrame(listaDeWorks)



# %%
# !! {"metadata":{
# !!   "id": "1kgROtf4W8wR",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 696
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286822911,
# !!     "user_tz": 180,
# !!     "elapsed": 16,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "3ec05b13-0d1b-439c-d4d0-281617eb9631"
# !! }}
argentina_works['affiliations'] = argentina_works['authorships'].apply(lambda x: [i['affiliations'] for i in x] if isinstance(x,list) else [])
print(argentina_works['affiliations'].head())
print(argentina_works.shape)
argentina_works.head()


# %%
# !! {"metadata":{
# !!   "id": "HaHJ_MaCwByA",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 335
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286824114,
# !!     "user_tz": 180,
# !!     "elapsed": 1214,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "4f4f5ad1-8524-4d0d-95ed-41cdd4d8200d"
# !! }}
argentina_works['raw_affiliation_string'] = argentina_works['affiliations'].apply(lambda x: [d.get('raw_affiliation_string') for sublist in x for d in sublist if isinstance(sublist, list) and isinstance(d, dict)] if isinstance(x, list) else [])
argentina_works['institution_ids']        = argentina_works['affiliations'].apply(lambda x: [d.get('institution_ids') for sublist in x for d in sublist if isinstance(sublist, list) and isinstance(d, dict)] if isinstance(x, list) else [])
argentina_works=argentina_works.explode('institution_ids')
argentina_works.head(2)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "ahRH88BgmfGn",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286824115,
# !!     "user_tz": 180,
# !!     "elapsed": 6,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "9900aae9-4897-4334-d5cf-3e95d60fefb3"
# !! }}
argentina_works=argentina_works[['id','institution_ids','raw_affiliation_string']]
print(argentina_works.head(2))


# %%
# !! {"metadata":{
# !!   "id": "wCUsTUzYs_np",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286825870,
# !!     "user_tz": 180,
# !!     "elapsed": 1760,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# Ensure that 'institution_ids' and 'raw_affiliation_string' have the same number of elements for each row.

# Option 1: Zip the two columns, filter out unequal length entries, then explode

# Create a new column with zipped values
argentina_works['zipped_data'] = argentina_works.apply(
    lambda row: list(zip(row['institution_ids'], row['raw_affiliation_string']))
    if len(row['institution_ids']) == len(row['raw_affiliation_string']) else [], axis=1
)
# Filter out rows with empty zipped_data (unequal length entries)
argentina_works = argentina_works[argentina_works['zipped_data'].apply(len) > 0]

# Explode the zipped column
argentina_works1 = argentina_works.explode('zipped_data')

# Split the zipped data back into original columns
argentina_works1[['institution_ids', 'raw_affiliation_string']] = pd.DataFrame(
    argentina_works1['zipped_data'].tolist(), index=argentina_works1.index
)


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 112
# !!   },
# !!   "id": "6yVFNnkytJ3T",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286825870,
# !!     "user_tz": 180,
# !!     "elapsed": 15,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "40a5b66a-0ea2-4a52-ff71-1a4a50d7211b"
# !! }}
argentina_works1.head(2)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "drAZajbdsdbv",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286825871,
# !!     "user_tz": 180,
# !!     "elapsed": 14,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "f99debf0-81c3-4166-e9a5-47cd51edadd3"
# !! }}
argentina_works1['works_id'] = argentina_works1['id'].apply(lambda x: x.split("/")[-1])
#argentina_works1['original_affiliation']=argentina_works1['raw_affiliation_string'].apply(lambda x: x.split("']")[0])
argentina_works1['institution_ids'] = argentina_works1['institution_ids'].apply(lambda x: x.split("/")[-1])
argentina_works1['institution_ids'] =argentina_works1['institution_ids'] .apply(lambda x: x.split("I")[-1])
argentina_works1['affiliation_id']=argentina_works1['institution_ids']
argentina_works1['original_affiliation']=argentina_works1['raw_affiliation_string']

print(argentina_works1.shape)
affiliations=argentina_works1
print(affiliations.head(2))

# %%
# !! {"metadata":{
# !!   "id": "6VVAQ7sT73Tp",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286825871,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "dE8evuBSz_hZ",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286827788,
# !!     "user_tz": 180,
# !!     "elapsed": 917,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "2f69a8d7-6751-4704-e056-6ad7f6c57161"
# !! }}
##agregado para armar dataset de prueba en testeo
print(affiliations.shape)

affiliation_muestra_para_testeo=affiliations.sample(frac=0.1)
affiliation_muestra_para_testeo.shape
affiliation_muestra_para_testeo.head()
affiliation_muestra_para_testeo=affiliation_muestra_para_testeo[['institution_ids','raw_affiliation_string','works_id']]
affiliation_muestra_para_testeo=affiliation_muestra_para_testeo.rename(columns={'institution_ids':'labels', 'works_id': 'paper_id', 'raw_affiliation_string': 'affiliation_string'})
print(affiliation_muestra_para_testeo.head())

affiliation_muestra_para_testeo.to_csv(f"{base_path}V2/002_Model/affiliations_para_testeo.csv")


# %%
# !! {"metadata":{
# !!   "id": "lhsCoWBnlmgX",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286827789,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "17b4b647-b3dc-4736-ceed-e80dde6eb1b8",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   }
# !! }}
affiliation_muestra_para_testeo.shape

# %%
# !! {"metadata":{
# !!   "id": "LTQsf3sv8ToG",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286827789,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
##nos quedamos con estas observaciones para affiliations
affiliations1=affiliations-affiliation_muestra_para_testeo

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "UebEDdYNYE2K",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286827790,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "e0e345c9-fb0b-4f46-f0b9-7773f84a9501"
# !! }}
affiliations1.shape

# %%
# !! {"metadata":{
# !!   "id": "tev_N9uAizOm",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286840008,
# !!     "user_tz": 180,
# !!     "elapsed": 12222,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
import os
os.chdir(f"{base_path}V2/002_Model/")

# Now you can safely write the parquet file
affiliations1.to_parquet('affiliations.parquet')

# Assuming 'spark' is your SparkSession object
affiliations = spark.read.parquet(f"{base_path}V2/002_Model/affiliations.parquet")


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "VmSxBUuWm3sw",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286842775,
# !!     "user_tz": 180,
# !!     "elapsed": 2772,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "ed7ee083-cec5-4e6b-a586-2df8323b560d"
# !! }}
print(affiliations.head())

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "18b42ff7-137e-46f4-a309-4fa30887954a",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "Y18gO-PYY81B"
# !! }}
"""
#### Getting ROR aff strings
"""

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {
# !!       "byteLimit": 2048000,
# !!       "rowLimit": 10000
# !!     },
# !!     "inputWidgets": {},
# !!     "nuid": "96d8d214-16a3-4090-93ba-485561cc6a56",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "6326WJ4tY81B",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286842775,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
dedup_affs = affiliations.select(F.trim(F.col('original_affiliation')).alias('original_affiliation'), 'affiliation_id')\
.filter(F.col('original_affiliation').isNotNull())\
.filter(F.col('original_affiliation')!='')\
.withColumn('aff_len', F.length(F.col('original_affiliation')))\
.filter(F.col('aff_len')>2)\
.groupby(['original_affiliation','affiliation_id']) \
.agg(F.count(F.col('affiliation_id')).alias('aff_string_counts'))

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {
# !!       "byteLimit": 2048000,
# !!       "rowLimit": 10000
# !!     },
# !!     "inputWidgets": {},
# !!     "nuid": "d53b5f75-9d2b-4b62-8f7d-7b1c0cce742a",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "Eua09x98Y81C",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286845148,
# !!     "user_tz": 180,
# !!     "elapsed": 2377,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "e1155d6c-f438-43b6-c3d4-52d4c632e6b2"
# !! }}
dedup_affs.cache().count()

# %%
# !! {"metadata":{
# !!   "id": "7oDcsZpVBrBX",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286845148,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "c6f62ea2-a94c-45b0-bdc3-cfc08d5b980d"
# !! }}
dedup_affs.head(n=3)

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "0185bff4-2d20-4fce-b3cf-be8fb41386cc",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "Hf6UxRrSY81C",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286848218,
# !!     "user_tz": 180,
# !!     "elapsed": 3074,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror_data = spark.read.parquet(f"{base_path}V2/001_Exploration/ror_strings.parquet") \
.select('original_affiliation','affiliation_id')

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "8280c348-e71f-4a44-ac11-b890500a5d02",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "AtlUPzcfY81D",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286850179,
# !!     "user_tz": 180,
# !!     "elapsed": 1965,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "799c6a31-e325-4578-daed-233b530d836b"
# !! }}
ror_data.cache().count()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "BEilrB2dmesq",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286850179,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "50244f79-f980-4583-cefe-287dc536958b"
# !! }}
ror_data.head(n=2)

# %%
# !! {"metadata":{
# !!   "id": "Sv_QTa0RY81D"
# !! }}
"""
### Gathering training data

Since we are looking at all institutions, we need to up-sample the institutions that don't have many affiliation strings and down-sample the institutions that have large numbers of strings. There was a balance here that needed to be acheived. The more samples that are taken for each institution, the more overall training data we will have and the longer our model will take to train.

However, more samples also means more ways of an institution showing up in an affiliation string.

The number of samples was set to 50 as it was determined this was a good optimization point based on affiliation string count distribution and time it would take to train the model.

However, unlike in V1 where we tried to keep all institutions at 50, for V2 we gave additional samples for institutions with more strings available. Specifically, we allowed those institutions to have up to 25 additional strings, for a total of 75.
"""

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "fa32b038-5cf3-4c8e-91de-f27c31865b0f",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "FnJKSJJhY81E",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286850179,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
num_samples_to_get = 50

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "da6aa4bc-419c-43dc-80c8-9c84c342f77d",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "Duj1ZxivY81E",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286850644,
# !!     "user_tz": 180,
# !!     "elapsed": 469,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
w1 = Window.partitionBy('affiliation_id')

filled_affiliations = dedup_affs \
    .join(ror_data.select('affiliation_id'), how='inner', on='affiliation_id') \
    .select('original_affiliation','affiliation_id') \
    .union(ror_data.select('original_affiliation','affiliation_id')) \
    .filter(~F.col('affiliation_id').isNull()) \
    .dropDuplicates() \
    .withColumn('random_prob', F.rand(seed=20)) \
    .withColumn('id_count', F.count(F.col('affiliation_id')).over(w1)) \
    .withColumn('scaled_count', F.lit(1)-((F.col('id_count') - F.lit(num_samples_to_get))/(F.lit(3500000) - F.lit(num_samples_to_get)))) \
    .withColumn('final_prob', F.col('random_prob')*F.col('scaled_count'))

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "1f742262-b1cf-4d85-8f53-a84a8803b723",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "b4zQisMdY81E",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286852847,
# !!     "user_tz": 180,
# !!     "elapsed": 2206,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "b7a1c38a-3a35-4b7f-91e9-39354901b6aa"
# !! }}
filled_affiliations.select('affiliation_id').distinct().count()


# %%
# !! {"metadata":{
# !!   "id": "4vryeAaSEHRo",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286854736,
# !!     "user_tz": 180,
# !!     "elapsed": 1893,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "9015bc1d-ba41-4fb0-ad92-345ea3bcff1f"
# !! }}
filled_affiliations.head(n=2)

# %%
# !! {"metadata":{
# !!   "id": "UddiLRxPER5v",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286854736,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "6f6dfa49-c78f-4c22-accd-2ce0df6bcfeb",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "ONvrkrtzY81F",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 130
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286856598,
# !!     "user_tz": 180,
# !!     "elapsed": 1865,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "c04a1bfe-6a06-49be-dbbe-a62ba45458db"
# !! }}
less_than = filled_affiliations.dropDuplicates(subset=['affiliation_id']).filter(F.col('id_count') < num_samples_to_get).toPandas()

less_than['affiliation_id'] = less_than['affiliation_id'].astype('str')
less_than['affiliation_id'] = less_than['affiliation_id'].apply(lambda x: x.split("/")[-1])

print(less_than.shape)
less_than.head(n=2)


# %%
# !! {"metadata":{
# !!   "id": "NNW5lQW_AjlB",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732286856598,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
temp_df_list = []
for aff_id in less_than['affiliation_id'].unique():
    temp_df = less_than[less_than['affiliation_id']==aff_id].copy()
    help_df = temp_df.sample(num_samples_to_get - temp_df.shape[0], replace=True)
    temp_df_list.append(pd.concat([temp_df, help_df], axis=0))
less_than_df = pd.concat(temp_df_list, axis=0)

# %%
# !! {"metadata":{
# !!   "id": "jCIJoZKXGVqE"
# !! }}
# only install fsspec and s3fs
less_than_df[['original_affiliation', 'affiliation_id']].to_parquet(f"{iteration_save_path}lower_than_{num_samples_to_get}.parquet")
w1 = Window.partitionBy('affiliation_id').orderBy('random_prob')

more_than = filled_affiliations.filter(F.col('id_count') >= num_samples_to_get) \
.withColumn('row_number', F.row_number().over(w1)) \
.filter(F.col('row_number') <= num_samples_to_get+25)

more_than.cache().count()


more_than=more_than.toPandas()

more_than['affiliation_id'] = more_than['affiliation_id'].astype('str')
more_than['affiliation_id'] = more_than['affiliation_id'].apply(lambda x: x.split("/")[-1])
more_than[['original_affiliation', 'affiliation_id']].to_parquet(f"{iteration_save_path}more_than_{num_samples_to_get}.parquet")


#more_than.select('original_affiliation', 'affiliation_id') \
#.coalesce(1).write.mode('overwrite').parquet(f"{iteration_save_path}more_than_{num_samples_to_get}")

# %%
# !! {"metadata":{
# !!   "id": "l1Jd2vR2PVDN"
# !! }}


# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "67ba033b-7145-4388-8467-7b46db44a94c",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "XSnRWfxzY81F"
# !! }}
less_than.sample(5)

# %%
# !! {"metadata":{
# !!   "application/vnd.databricks.v1+cell": {
# !!     "cellMetadata": {},
# !!     "inputWidgets": {},
# !!     "nuid": "5443d79e-8ae0-468f-98c1-6ff31bdc26ac",
# !!     "showTitle": false,
# !!     "title": ""
# !!   },
# !!   "id": "MeWkbarCY81I"
# !! }}
less_than_df.shape

# %%
# !! {"metadata":{
# !!   "id": "TKwwprZZlLKl"
# !! }}
more_than.head()

# %%
# !! {"metadata":{
# !!   "id": "Py2otP1tQoV1"
# !! }}
more_than.shape

# %%
# !! {"metadata":{
# !!   "id": "U_HJ4jePUrq4"
# !! }}
sub_p_res = subprocess.run(['pip', 'install', 'colab-convert'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['colab-convert', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/001_get_training_data_SPARK.ipynb', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/001_get_training_data_SPARK.p'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>


# %%
# !! {"main_metadata":{
# !!   "application/vnd.databricks.v1+notebook": {
# !!     "dashboards": [],
# !!     "language": "python",
# !!     "notebookMetadata": {
# !!       "pythonIndentUnit": 4
# !!     },
# !!     "notebookName": "institutional_affiliation_classification_V25_all_strings",
# !!     "notebookOrigID": 1621990259572172,
# !!     "widgets": {}
# !!   },
# !!   "kernelspec": {
# !!     "display_name": "Python 3",
# !!     "name": "python3"
# !!   },
# !!   "language_info": {
# !!     "codemirror_mode": {
# !!       "name": "ipython",
# !!       "version": 3
# !!     },
# !!     "file_extension": ".py",
# !!     "mimetype": "text/x-python",
# !!     "name": "python",
# !!     "nbconvert_exporter": "python",
# !!     "pygments_lexer": "ipython3",
# !!     "version": "3.8.12"
# !!   },
# !!   "colab": {
# !!     "provenance": [],
# !!     "gpuType": "T4"
# !!   },
# !!   "accelerator": "GPU"
# !! }}
