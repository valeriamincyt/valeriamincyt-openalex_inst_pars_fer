# %%
# !! {"metadata":{
# !!   "id":"cc-imports"
# !! }}

#<cc-imports>

#import subprocess

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "ozObisEqpml7",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721842631,
# !!     "user_tz": 180,
# !!     "elapsed": 38230,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "9ff9f4d0-a48a-4b5c-d6bb-0a613ff03166"
# !! }}
import pickle
import json
import json,urllib.request
#sub_p_res = subprocess.run(['pip', 'install', 'redshift_connector'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
#import redshift_connector
import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
import numpy as np
import re
import os
#sub_p_res = subprocess.run(['pip', 'install', 'unidecode'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
from unidecode import unidecode
from collections import Counter
from math import ceil
#sub_p_res = subprocess.run(['pip', 'install', 'langdetect'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
from langdetect import detect
from random import sample

#from google.colab import drive

#sub_p_res = subprocess.run(['pip', 'install', 'pyalex'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
import pyalex
from pyalex import Works, Authors, Sources, Institutions, Topics, Publishers, Funders
from pyalex import config

#sub_p_res = subprocess.run(['pip', 'install', 'colab-convert'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>


# %%
# !! {"metadata":{
# !!   "id": "06735fac"
# !! }}
"""
## Support Files
"""

# %%
# !! {"metadata":{
# !!   "id": "7f719469"
# !! }}
"""
Throughout the modeling process, some of the model artifacts needed to be updated and so this notebook was used to quickly update those files.
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "55qdRS8_dzOu",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721884932,
# !!     "user_tz": 180,
# !!     "elapsed": 42307,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "332d927b-7727-4c53-b5ef-92288020affd"
# !! }}
## comentar siguiente línea en script
drive.mount('/content/drive')

# %%
# !! {"metadata":{
# !!   "id": "7TTi6GuynmN0"
# !! }}
"""
**Cambiar path**
"""

# %%
# !! {"metadata":{
# !!   "id": "unZF4Ce5ZHTZ",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721884932,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
base_path = '/content/drive/MyDrive/openalex-institution-parsing/'
base_path = '../../../'
# location where current files are located
curr_model_artifacts_location = os.path.join(base_path, 'Crudos/institution_tagger_v2_artifacts/')


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "7325fe49",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889530,
# !!     "user_tz": 180,
# !!     "elapsed": 4602,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "f9cc5ba5-7968-40cb-a973-16b58b8af74d"
# !! }}


# Load the needed files
with open(f"{curr_model_artifacts_location}departments_list.pkl", "rb") as f:
    departments_list = pickle.load(f)
print("Loaded list of departments")

with open(f"{curr_model_artifacts_location}countries_list_flat.pkl", "rb") as f:
    countries_list_flat = pickle.load(f)
print("Loaded flat list of countries")

with open(f"{curr_model_artifacts_location}countries.json", "r") as f:
    countries_dict = json.load(f)
print("Loaded countries dictionary")

with open(f"{curr_model_artifacts_location}city_country_list.pkl", "rb") as f:
    city_country_list = pickle.load(f)
print("Loaded strings of city/country combinations")

# %%
# !! {"metadata":{
# !!   "id": "5502b0dd"
# !! }}
"""
### Looking at ROR
"""

# %%
# !! {"metadata":{
# !!   "id": "c49d9de9",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889530,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_geoname_admin(address_dict):
    try:
        geoname_admin = address_dict['geonames_city']['geonames_admin1']['name']
    except:
        geoname_admin = "None"

    return geoname_admin

# %%
# !! {"metadata":{
# !!   "id": "1612ec85",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889530,
# !!     "user_tz": 180,
# !!     "elapsed": 12,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_final_region(ror_state, ror_region):
    if isinstance(ror_state, str):
        return ror_state
    elif isinstance(ror_region, str):
        return ror_region
    else:
        return None

# %%
# !! {"metadata":{
# !!   "id": "05bbef34",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889530,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def check_for_backwards_univ(curr_names):
    names = curr_names.copy()
    for one_name in curr_names:
        split_name = one_name.split(" ")
        if len(split_name) == 3:
            if (split_name[0] == 'University') & (split_name[1] == 'of'):
                names.append(f"{split_name[2]} University")
        elif len(split_name) == 2:
            if (split_name[1] == 'University'):
                names.append(f"University of {split_name[0]}")
        else:
            pass
    return names

# %%
# !! {"metadata":{
# !!   "id": "05b65ea6",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889530,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def add_names_to_list(all_names):
    names = all_names.copy()
    if "Harvard University" in names:
        names.append("Harvard Medical School")
    elif "University of Oxford" in names:
        names.append("Oxford University")
    else:
        pass

    return names

# %%
# !! {"metadata":{
# !!   "id": "4745c785",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889531,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_exact_names(name, aliases, acronyms, labels):
    all_names = [name] + aliases + acronyms + [i['label'] for i in labels]
    all_names = add_names_to_list(all_names)
    all_names = [x for x in all_names if ~x.startswith('Department of')]
    all_names_clean = [string_match_clean(x) for x in all_names]
    return [x for x in all_names_clean if len(x) > 4]

# %%
# !! {"metadata":{
# !!   "id": "efc67f40",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889531,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "m9cxadr8Rt6n"
# !! }}
"""

"""

# %%
# !! {"metadata":{
# !!   "id": "181903bd",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889531,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def string_match_clean(text):
    #replace "&" with "and"
    if "r&d" not in text.lower():
        text = text.replace(" & ", " and ")

    # take country out
    if text.strip().endswith(")"):
        for country in countries_list_flat:
            if text.strip().endswith(f"({country})"):
                text = text.replace(f"({country})", "")

    # use unidecode
    text = unidecode(text.strip())

    # replacing common abbreviations
    text = text.replace("Univ.", "University")
    text = text.replace("Lab.", "Laboratory")

    # take out spaces, commas, dashes, periods, etcs
    text = re.sub("[^0-9a-zA-Z]", "", text)

    return text

# %%
# !! {"metadata":{
# !!   "id": "f91cc78e",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889531,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def list_of_all_names(oa_name, ror_names, extra_names, use_extra_names=False):
    banned_names = ['UniversityHospital','Coastal','Brunswick','Continental']
    if isinstance(ror_names, list):
        pass
    else:
        ror_names = []

    if isinstance(oa_name, str):
        oa_string = [string_match_clean(oa_name)]
    else:
        oa_string = []

    if use_extra_names:
        if isinstance(extra_names, list):
            pass
        else:
            extra_names = []
    else:
        extra_names = []

    return [x for x in list(set(oa_string+ror_names+extra_names)) if
            ((len(x) > 4) & (x not in banned_names))]

# %%
# !! {"metadata":{
# !!   "id": "afb19811",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721889531,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# this file is not provided but the needed data is all institutions in OpenAlex
# with the following columns: 'ror_id','affiliation_id','display_name'
institutions_df = pd.read_parquet(f'{base_path}Crudos/OA_static_institutions_single_file.parquet')

### ALTERNATIVA DE OA_static_institutions_single_file.parquet bajando desde Python
"""
pyalex.config.email = "mfartola.mincyt@gmail.com"

config.max_retries = 0
config.retry_backoff_factor = 0.1
config.retry_http_codes = [429, 500, 503]
"""
# %%
# !! {"metadata":{
# !!   "id": "e6ahXrT8CVyq",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721894134,
# !!     "user_tz": 180,
# !!     "elapsed": 4610,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "087f0266-a992-4c7c-8a93-814946fbfc79"
# !! }}
"""
data = urllib.request.urlopen("https://api.openalex.org/institutions?filter=country_code:AR&per-page=200&page=1").read()
data1 = urllib.request.urlopen("https://api.openalex.org/institutions?filter=country_code:AR&per-page=200&page=2").read()

output = json.loads(data)
output1 = json.loads(data1)

argentina1=pd.DataFrame(output['results'])
argentina2=pd.DataFrame(output1['results'])
argentina=pd.concat([argentina1,argentina2])
argentina=pd.DataFrame(argentina)
print(argentina.head())
argentina.shape
institutions_df=argentina
institutions_df.columns
"""
# %%
# !! {"metadata":{
# !!   "id": "3e748631-c2cc-4b84-80c4-d68d43f6fe38",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721894586,
# !!     "user_tz": 180,
# !!     "elapsed": 464,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 1000
# !!   },
# !!   "outputId": "1c1ce11e-48d8-487f-b149-dcca8af5ed32"
# !! }}
institutions_df['affiliation_id'] = institutions_df['id'].apply(lambda x: x.split("/")[-1])
institutions_df['affiliation_id']=institutions_df['affiliation_id'].apply(lambda x: x.split("I")[-1])
#institutions_df['affiliation_id']=institutions_df['affiliation_id'].astype(int)  ### paso los ids como enteros

institutions_df['ror_id'] = institutions_df['ror'].apply(lambda x: x.split("/")[-1])

#institutions_df['display_name'] = institutions_df['display_name_alternatives'] ###ojo, lo comenté!

print(institutions_df.head(1))

# %%
# !! {"metadata":{
# !!   "id": "fKonaTPZdcjI",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721894588,
# !!     "user_tz": 180,
# !!     "elapsed": 14,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "d6e325be-357a-461e-fe77-b5879c2be8dd"
# !! }}
institutions_df.columns

# %%
# !! {"metadata":{
# !!   "id": "gLQlAmWGhhd9",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721894588,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 112
# !!   },
# !!   "outputId": "2abc6359-77eb-477e-9bee-1bdfe60030d0"
# !! }}
institutions_df=institutions_df[['affiliation_id','display_name','ror_id']]
print(institutions_df.head(2))

# %%
# !! {"metadata":{
# !!   "id": "df229664",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721913463,
# !!     "user_tz": 180,
# !!     "elapsed": 18885,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "7d0526bf-dabf-4497-ee7f-a81f9325b0a7"
# !! }}
##Carga de ror

file_path = os.path.join(base_path, 'Crudos/v1.53-2024-10-14-ror-data.json')

ror = pd.read_json(file_path)
ror.head(2)
ror.columns

# %%
# !! {"metadata":{
# !!   "id": "c9a524c5",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721913463,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror['address'] = ror['addresses'].apply(lambda x: x[0])
ror['ror_id'] = ror['id'].apply(lambda x: x.split("/")[-1])
ror['ror_id']=ror['ror_id'].apply(lambda x: x.split("I")[-1])

#ror['types'] = ror['types'].apply(lambda x: x[0])
ror['types'] = ror['types'].apply(lambda x: x[0] if len(x)>0 else "")
##
#ror[ror['types'].isnull()]

# %%
# !! {"metadata":{
# !!   "id": "04edf601",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721913965,
# !!     "user_tz": 180,
# !!     "elapsed": 506,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror['country_code'] = ror['country'].apply(lambda x: x['country_code'])
ror['country_name'] = ror['country'].apply(lambda x: x['country_name'])
ror['city'] = ror['address'].apply(lambda x: x['city'])
ror['state'] = ror['address'].apply(lambda x: x['state'])
ror['region'] = ror['address'].apply(get_geoname_admin)

# %%
# !! {"metadata":{
# !!   "id": "76SZlQ8meQOS",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721913966,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "0f4e4a10-968a-4993-f94c-ae1d2429c356"
# !! }}
ror.columns

# %%
# !! {"metadata":{
# !!   "id": "fced5152",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721913966,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror_to_join = ror[['ror_id','name','status','types','aliases','acronyms','labels','city',
                   'state','region','country_name']].copy()

ror_to_join.columns = ['ror_id','name','status','types','aliases','acronyms','labels','city',
                       'temp_state','temp_region','country']

# %%
# !! {"metadata":{
# !!   "id": "8926d655",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916303,
# !!     "user_tz": 180,
# !!     "elapsed": 2342,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 81
# !!   },
# !!   "outputId": "187171c8-9520-43e4-b6c2-a48fcdf40ebd"
# !! }}
ror_to_join['state'] = ror_to_join.apply(lambda x: get_final_region(x.temp_state, x.temp_region), axis=1)
ror_to_join.head(1)

# %%
# !! {"metadata":{
# !!   "id": "UG8_srRZg6rs",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916303,
# !!     "user_tz": 180,
# !!     "elapsed": 31,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 81
# !!   },
# !!   "outputId": "abf997dd-e753-4ba7-dc46-b2ee1d2fd8ea"
# !! }}
institutions_df.head(1)

# %%
# !! {"metadata":{
# !!   "id": "vYEOW1sSg_mp",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916303,
# !!     "user_tz": 180,
# !!     "elapsed": 29,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "577d4153",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916304,
# !!     "user_tz": 180,
# !!     "elapsed": 29,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "509d326d-18dc-4c2e-af34-678c4be5a507"
# !! }}
inst_ror = ror_to_join.merge(institutions_df[['ror_id','affiliation_id','display_name']],how='inner', on='ror_id')
inst_ror.shape

# %%
# !! {"metadata":{
# !!   "id": "GbiplSWsuizF",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916304,
# !!     "user_tz": 180,
# !!     "elapsed": 24,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 195
# !!   },
# !!   "outputId": "c91d3adc-47da-49ae-8f51-50bf57edb4c2"
# !! }}
inst_ror.head(n=3)

# %%
# !! {"metadata":{
# !!   "id": "f03e628c"
# !! }}
"""
#### Getting file of multi-institution names
"""

# %%
# !! {"metadata":{
# !!   "id": "de54de85",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916304,
# !!     "user_tz": 180,
# !!     "elapsed": 21,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
multi_inst_names_df = ror_to_join[['ror_id','name']].merge(institutions_df[['ror_id','affiliation_id']],
                                                        how='left', on='ror_id') \
['name'].value_counts().reset_index()

# %%
# !! {"metadata":{
# !!   "id": "ac4cc44d",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916304,
# !!     "user_tz": 180,
# !!     "elapsed": 20,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 363
# !!   },
# !!   "outputId": "d4a7cf6b-bc6c-47a9-d30b-50c194a1aa2e"
# !! }}
multi_inst_names_df.head(10)

# %%
# !! {"metadata":{
# !!   "id": "6a8ed7bd",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916304,
# !!     "user_tz": 180,
# !!     "elapsed": 19,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
multi_inst_names = multi_inst_names_df[multi_inst_names_df['count']>1]
multi_inst_names
multi_inst_names=multi_inst_names['name'].tolist()
#multi_inst_names

# %%
# !! {"metadata":{
# !!   "id": "a4b26562",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916305,
# !!     "user_tz": 180,
# !!     "elapsed": 19,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
## paso a entero a los ids antes de pasarlo a lista
multi_inst_names_ids = inst_ror[inst_ror['name'].isin(multi_inst_names)]['affiliation_id'].astype(int).tolist()

# %%
# !! {"metadata":{
# !!   "id": "cb8a31d6",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916695,
# !!     "user_tz": 180,
# !!     "elapsed": 408,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
with open(f"{curr_model_artifacts_location}multi_inst_names_ids1.pkl", "wb") as f:
    pickle.dump(multi_inst_names_ids, f)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "Ilayr3oCXDSs",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916695,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "30e98702-cd9a-4a8b-b464-3a9d5223969c"
# !! }}
multi_inst_names_ids

# %%
# !! {"metadata":{
# !!   "id": "514951ad"
# !! }}
"""
### Getting Mapping of Inactive Institutions
"""

# %%
# !! {"metadata":{
# !!   "id": "a8eaf926"
# !! }}
"""
There are institutions in ROR that are listed as "Withdrawn" or "Inactive". There was some thought to use the old data associated with these ROR IDs and apply them to successors but for this model, we decided to hold off on doing this because we were unsure if there would be a benefit to doing so. Therefore, the code is provided but this data was not used in building the model.
"""

# %%
# !! {"metadata":{
# !!   "id": "01e97850",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916695,
# !!     "user_tz": 180,
# !!     "elapsed": 6,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_successors_from_relationships(relationships):
    successors = []
    parents = []
    for relationship in relationships:
        if relationship['type'] == 'Successor':
            successors.append(relationship['id'].split("/")[-1])
        elif relationship['type'] == 'Parent':
            parents.append(relationship['id'].split("/")[-1])
        else:
            pass
    return [successors, parents]

# %%
# !! {"metadata":{
# !!   "id": "cead5a33",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721916695,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_extra_names(ror_id):
    if ror_id in successor_dict.keys():
        extra_names = []
        for old_id in successor_dict[ror_id]['ror_id']:
            extra_names += old_name_data[old_id]['successor_names']

        extra_names = list(set(extra_names))
    else:
        extra_names = []

    return extra_names

# %%
# !! {"metadata":{
# !!   "id": "ca19566a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917119,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "2fa35999-958c-4680-fd8a-164177a9e74c"
# !! }}
withdrawn_or_inactive_df = ror[ror['status'].isin(['withdrawn','inactive'])].copy()
withdrawn_or_inactive_df.shape

# %%
# !! {"metadata":{
# !!   "id": "b8fce951",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917120,
# !!     "user_tz": 180,
# !!     "elapsed": 12,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
withdrawn_or_inactive_df['successors_parents'] = withdrawn_or_inactive_df['relationships'] \
    .apply(get_successors_from_relationships)

# %%
# !! {"metadata":{
# !!   "id": "bdadfc74",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917120,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
withdrawn_or_inactive_df['successors'] = withdrawn_or_inactive_df['successors_parents'].apply(lambda x: x[0])
withdrawn_or_inactive_df['parents'] = withdrawn_or_inactive_df['successors_parents'].apply(lambda x: x[1])

# %%
# !! {"metadata":{
# !!   "id": "46623b5c",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917120,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
withdrawn_or_inactive_df['successor_len'] = withdrawn_or_inactive_df['successors'].apply(len)

# %%
# !! {"metadata":{
# !!   "id": "c59d7901",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917120,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "6cee3c88-1eb4-4d67-dcf0-6413a6a4b068"
# !! }}
to_add_to_successors = withdrawn_or_inactive_df[withdrawn_or_inactive_df['successor_len']==1].copy()
to_add_to_successors['successor'] = to_add_to_successors['successors'].apply(lambda x: x[0])
to_add_to_successors.shape

# %%
# !! {"metadata":{
# !!   "id": "192b8697",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917121,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
to_add_to_successors['successor_names'] = to_add_to_successors.apply(lambda x: get_exact_names(x['name'],
                                                                                               x.aliases,
                                                                                               x.acronyms,
                                                                                               x.labels),
                                                                     axis=1)

# %%
# !! {"metadata":{
# !!   "id": "a084702a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917121,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
old_name_data = to_add_to_successors.set_index('ror_id')[['successor_names']].to_dict(orient='index')

# %%
# !! {"metadata":{
# !!   "id": "37934f5f",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917121,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
successor_dict = to_add_to_successors.groupby('successor')['ror_id'].apply(list).reset_index()\
    .set_index('successor').to_dict(orient='index')

# %%
# !! {"metadata":{
# !!   "id": "45abfc2c"
# !! }}
"""
### Getting ROR String Matching File and Affiliation Dictionary
"""

# %%
# !! {"metadata":{
# !!   "id": "3564dce5",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917624,
# !!     "user_tz": 180,
# !!     "elapsed": 36,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 458
# !!   },
# !!   "outputId": "5909ae71-1b89-499a-fe87-d81bff6b187f"
# !! }}
inst_ror['extra_names'] = inst_ror['ror_id'].apply(get_extra_names)
inst_ror['extra_names']


# %%
# !! {"metadata":{
# !!   "id": "d7dbd006-7fb0-42f2-9b81-1f2c28855861",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917624,
# !!     "user_tz": 180,
# !!     "elapsed": 34,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "a7965af6-5f21-4d8e-b5cf-cd7b14b9ee2f",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917624,
# !!     "user_tz": 180,
# !!     "elapsed": 34,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 327
# !!   },
# !!   "outputId": "cfba2bc9-1ffc-4801-b456-091bf20d87d6"
# !! }}
inst_ror.head()

# %%
# !! {"metadata":{
# !!   "id": "99749672",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917625,
# !!     "user_tz": 180,
# !!     "elapsed": 33,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
inst_ror['exact_names'] = (inst_ror.apply(lambda x: get_exact_names(x['name'], x.aliases,
                                                                         x.acronyms, x.labels), axis=1))

# %%
# !! {"metadata":{
# !!   "id": "84343224-9f6c-4810-ae4a-f8f24b5ba74e",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917625,
# !!     "user_tz": 180,
# !!     "elapsed": 32,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 458
# !!   },
# !!   "outputId": "5611b211-f6ec-4f1a-83a9-148e332a5679"
# !! }}
inst_ror.apply(lambda x: get_exact_names(x['name'], x.aliases,x.acronyms, x.labels), axis=1)

# %%
# !! {"metadata":{
# !!   "id": "4b3f3465",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917625,
# !!     "user_tz": 180,
# !!     "elapsed": 31,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
inst_ror['final_names'] = inst_ror.apply(lambda x: list_of_all_names(x.display_name, x.exact_names,
                                                                     x.extra_names,
                                                                     use_extra_names=False), axis=1)

# %%
# !! {"metadata":{
# !!   "id": "92a2d20a-18c1-4027-aa7b-3d8ec118d112",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917625,
# !!     "user_tz": 180,
# !!     "elapsed": 30,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 458
# !!   },
# !!   "outputId": "6ace482e-ba1e-4a14-ad58-b689faae7e7f"
# !! }}
inst_ror.apply(lambda x: list_of_all_names(x.display_name, x.exact_names,
                                                                     x.extra_names,
                                                                     use_extra_names=False), axis=1)

# %%
# !! {"metadata":{
# !!   "id": "R8FiajM-YQsH",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917626,
# !!     "user_tz": 180,
# !!     "elapsed": 30,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
## lo paso a entero antes de hacerlo indice
inst_ror['affiliation_id']=inst_ror['affiliation_id'].astype(int)

# %%
# !! {"metadata":{
# !!   "id": "b3d11044",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917626,
# !!     "user_tz": 180,
# !!     "elapsed": 29,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
new_affiliation_dict = inst_ror.set_index('affiliation_id')[['display_name','city','state',
                                                             'country','final_names','ror_id','types']] \
.to_dict(orient='index')

# %%
# !! {"metadata":{
# !!   "id": "B_Gg0uEWeAlR",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721917626,
# !!     "user_tz": 180,
# !!     "elapsed": 28,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "85f709b8-cb76-4671-e35c-9c1417290d16"
# !! }}
new_affiliation_dict

# %%
# !! {"metadata":{
# !!   "id": "51455a6c",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721918721,
# !!     "user_tz": 180,
# !!     "elapsed": 1113,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
with open(f"{curr_model_artifacts_location}full_affiliation_dict1.pkl", "wb") as f:
    pickle.dump(new_affiliation_dict, f)

# %%
# !! {"metadata":{
# !!   "id": "116d227f",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721918721,
# !!     "user_tz": 180,
# !!     "elapsed": 74,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "aa94956d-ee0c-4581-c530-80db3813ab9f"
# !! }}
new_affiliation_dict

# %%
# !! {"metadata":{
# !!   "id": "14ab1cb0"
# !! }}
"""
### Updating the city/country file
"""

# %%
# !! {"metadata":{
# !!   "id": "cc50efa2"
# !! }}
"""
This file is used to check the affiliation string to make sure it doesn't exactly match up with a city/region/country combo with no additional information.
"""

# %%
# !! {"metadata":{
# !!   "id": "9828eab8",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721918722,
# !!     "user_tz": 180,
# !!     "elapsed": 68,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "8e5cc953-51ed-4a38-f47b-bab8aac5882b"
# !! }}
city_region_country = inst_ror.drop_duplicates(subset=['city','country']).copy()
city_region_country.shape

# %%
# !! {"metadata":{
# !!   "id": "2bcdc598",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721918722,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
new_city_country_list = list(set([f"{i}{j}" for i,j in zip(city_region_country['city'].tolist(),
                                   city_region_country['country'].tolist())] +
         [f"{i}{j}{k}"for i,j,k in zip(city_region_country['city'].tolist(),
                                             city_region_country['state'].tolist(),
                                             city_region_country['country'].tolist()) if j ] +
         [f"{i}{j}" for i,j in zip(city_region_country['state'].tolist(),
                                   city_region_country['country'].tolist()) if i] +
         [f"{i}" for i in city_region_country['country'].tolist()] +
         [f"{i}" for i in city_region_country['state'].tolist() if i]))

new_city_country_list = list(set([string_match_clean(x) for x in new_city_country_list]))

# %%
# !! {"metadata":{
# !!   "id": "2f5b3da4",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721918722,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "5f7494fa-9002-4178-f22b-3b93cc253206"
# !! }}
len(new_city_country_list)

# %%
# !! {"metadata":{
# !!   "id": "993877b2",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721919113,
# !!     "user_tz": 180,
# !!     "elapsed": 395,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
with open(f"{curr_model_artifacts_location}city_country_list1.pkl", "wb") as f:
    pickle.dump(new_city_country_list, f)

# %%
# !! {"metadata":{
# !!   "id": "09c98435"
# !! }}
"""
### Flat country file is up to date
"""

# %%
# !! {"metadata":{
# !!   "id": "a16d04e3"
# !! }}
"""
Flat country file is needed to search for country in the string for the model.
"""

# %%
# !! {"metadata":{
# !!   "id": "3cc540d0",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721919500,
# !!     "user_tz": 180,
# !!     "elapsed": 391,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "dcd9253b-4dfd-4feb-9afb-611d347f2464"
# !! }}
len(list(set(countries_list_flat)))

# %%
# !! {"metadata":{
# !!   "id": "44847d44",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721919501,
# !!     "user_tz": 180,
# !!     "elapsed": 390,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
all_countries = []
for i in countries_dict.values():
    all_countries += i

# %%
# !! {"metadata":{
# !!   "id": "2189b26b",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721919501,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "261e1f9f-72d0-467a-e2c8-489eb5d79f29"
# !! }}
len(list(set(all_countries)))

# %%
# !! {"metadata":{
# !!   "id": "f5c0501f",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721920321,
# !!     "user_tz": 180,
# !!     "elapsed": 823,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
with open(f"{curr_model_artifacts_location}countries_list_flat1.pkl", "wb") as f:
    pickle.dump(list(set(all_countries)), f)

# %%
# !! {"metadata":{
# !!   "id": "979a7409"
# !! }}
"""
### Departments list update
"""

# %%
# !! {"metadata":{
# !!   "id": "c6eb8b28"
# !! }}
"""
Takes the old department list and updates it with additional department names.
"""

# %%
# !! {"metadata":{
# !!   "id": "05aeff8e",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721922820,
# !!     "user_tz": 180,
# !!     "elapsed": 2502,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
file_path = os.path.join(base_path, 'V2/001_Exploration/ror_string_beginnings/Education_dept')
with open(file_path, 'r') as f:
    education_dept_begs = f.readlines()

education_dept_begs = list(set([x.rstrip('\n') for x in education_dept_begs]))

# %%
# !! {"metadata":{
# !!   "id": "69dccd71",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721922821,
# !!     "user_tz": 180,
# !!     "elapsed": 6,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
departments_list = ['Psychology','Nephrology','Other departments','Other Departments','Nursing & Midwifery',
                    'Literature and Creative Writing','Neuroscience','Engineering','Computer Science',
                    'Chemistry','Biology','Medicine']

# %%
# !! {"metadata":{
# !!   "id": "7a9fbb2a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721922821,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
new_departments_list = list(set(departments_list + education_dept_begs))

# %%
# !! {"metadata":{
# !!   "id": "d918c68b",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721923610,
# !!     "user_tz": 180,
# !!     "elapsed": 794,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
with open(f"{curr_model_artifacts_location}departments_list1.pkl", "wb") as f:
    pickle.dump(new_departments_list, f)

# %%
# !! {"metadata":{
# !!   "id": "b1331282",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721923610,
# !!     "user_tz": 180,
# !!     "elapsed": 2,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "e0d69a31"
# !! }}
"""
### Make affiliation IDs integers
"""

# %%
# !! {"metadata":{
# !!   "id": "dc442f94",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721931408,
# !!     "user_tz": 180,
# !!     "elapsed": 7800,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "output_embedded_package_id": "1tPtIG246r28hgq-OmkiCN_Y34ABnST25"
# !!   },
# !!   "outputId": "50811f97-4920-4451-d8c2-ee2211fe1d7e"
# !! }}
with open(f"{curr_model_artifacts_location}affiliation_vocab.pkl", "rb") as f:
    affiliation_vocab_basic = pickle.load(f)

print(affiliation_vocab_basic)
print("Loaded affiliation vocab basic")


new_affiliation_vocab_basic = {int(i):int(j) for j,i in affiliation_vocab_basic.items()}
print(new_affiliation_vocab_basic)
print("New vocab basic")


# %%
# !! {"metadata":{
# !!   "id": "cc4147d2",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721935651,
# !!     "user_tz": 180,
# !!     "elapsed": 4247,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
with open(f"{curr_model_artifacts_location}new_affiliation_vocab1.pkl", "wb") as f:
    pickle.dump(new_affiliation_vocab_basic, f)

# %%
# !! {"metadata":{
# !!   "id": "12735c2b",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721935651,
# !!     "user_tz": 180,
# !!     "elapsed": 703,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "066c7a8d-5399-4472-93c2-874c6779f8a5"
# !! }}
print(new_affiliation_vocab_basic)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "PZpKYoe2TCtJ",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732721935651,
# !!     "user_tz": 180,
# !!     "elapsed": 675,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "47b17b4a-b0e8-41fc-c4bd-b672cb3e5c2a"
# !! }}
##comentar siguiente línea en script
#sub_p_res = subprocess.run(['colab-convert', '/content/drive/MyDrive/openalex-institution-parsing/V2/001_Exploration/Notebooks/002_Making_Model_Files_Fer.ipynb', '/content/drive/MyDrive/openalex-institution-parsing/V2/001_Exploration/Notebooks/002_Making_Model_Files_Fer.p'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>


# %%
# !! {"main_metadata":{
# !!   "kernelspec": {
# !!     "display_name": "Python 3 (ipykernel)",
# !!     "language": "python",
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
# !!     "version": "3.12.4"
# !!   },
# !!   "colab": {
# !!     "provenance": []
# !!   }
# !! }}
print('FINALIZADO OK')