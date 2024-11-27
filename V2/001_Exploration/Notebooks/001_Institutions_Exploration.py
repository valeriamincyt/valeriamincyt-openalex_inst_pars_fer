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
# !!   "id": "efc67f40",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285182506,
# !!     "user_tz": 180,
# !!     "elapsed": 30474,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "271effa5-8dee-4c31-bd4e-601b7b62bf43"
# !! }}
import pickle
import json
#sub_p_res = subprocess.run(['pip', 'install', 'redshift_connector'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
#import redshift_connector
import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
import numpy as np
#sub_p_res = subprocess.run(['pip', 'install', 'unidecode'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
import unidecode
import re

from collections import Counter
from math import ceil
#sub_p_res = subprocess.run(['pip', 'install', 'langdetect'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
from langdetect import detect
from random import sample

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "5Mv4YG9y1FEA",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285224926,
# !!     "user_tz": 180,
# !!     "elapsed": 42428,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "d40806ba-06fb-4924-ba89-79c8cd883f5c"
# !! }}
#from google.colab import drive
#drive.mount('/content/drive')

# %%
# !! {"metadata":{
# !!   "id": "3gjVHsMnp3BK"
# !! }}
"""
**Cambiar path para correr desde otro lado**
"""

# %%
# !! {"metadata":{
# !!   "id": "D2ICzsn7UR6V",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285224926,
# !!     "user_tz": 180,
# !!     "elapsed": 6,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
import os
#base_path = '/content/drive/MyDrive/openalex-institution-parsing/'
base_path = '../../../'

# %%
# !! {"metadata":{
# !!   "id": "5502b0dd"
# !! }}
"""
## Exploring the ROR Data to Create Artificial Training Data
"""

# %%
# !! {"metadata":{
# !!   "id": "df229664",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285242755,
# !!     "user_tz": 180,
# !!     "elapsed": 17834,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# Data was downloaded from the ROR website for the date seen in the file string below
# https://ror.readme.io/docs/data-dump
#ror = pd.read_json("./v1.19-2023-02-16-ror-data.json")

ror=pd.read_json(f'{base_path}Crudos/v1.53-2024-10-14-ror-data.json')

# %%
# !! {"metadata":{
# !!   "id": "c9a524c5",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285243039,
# !!     "user_tz": 180,
# !!     "elapsed": 288,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror['alias_len'] = ror['aliases'].apply(len)
ror['acronyms_len'] = ror['acronyms'].apply(len)
ror['labels_len'] = ror['labels'].apply(len)
ror['address_len'] = ror['addresses'].apply(len)
ror['address'] = ror['addresses'].apply(lambda x: x[0])
ror['ror_id'] = ror['id'].apply(lambda x: x.split("/")[-1])
#ror['types'] = ror['types'].apply(lambda x: x[0])
ror['types'] = ror['types'].apply(lambda x: x[0] if len(x)>0 else "")


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 396
# !!   },
# !!   "id": "6ee9e0fe",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285243040,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "52546d38-9c1e-451c-fe22-cd17834d3361"
# !! }}
ror[ror['ror_id']=='05kxf7578']

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "651ff1b1",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285245712,
# !!     "user_tz": 180,
# !!     "elapsed": 2677,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "1df00372-5021-49ac-e71d-3b6b8d9a70fc"
# !! }}
# this file is not provided but the needed data is all institutions in OpenAlex
# with the following columns: 'ror_id','affiliation_id'
#insts = pd.read_parquet("OA_static_institutions_single_file.parquet",
 #                       columns=['affiliation_id','ror_id'])


##Alternativa: bajando el archivo utilizando la API y colocando el link de la búsqueda

import json,requests
data = requests.get("https://api.openalex.org/institutions?filter=country_code:AR&per-page=200&page=1").text
data1 = requests.get("https://api.openalex.org/institutions?filter=country_code:AR&per-page=200&page=2").text

output = json.loads(data)
output1 = json.loads(data1)

argentina1=pd.DataFrame(output['results'])
argentina2=pd.DataFrame(output1['results'])
argentina=pd.concat([argentina1,argentina2])
argentina.head()
argentina.shape
insts=argentina
####
####
insts['ror_id'] = insts['ror'].apply(lambda x: x.split("/")[-1])
insts['affiliation_id'] = insts['id'].apply(lambda x: x.split("/")[-1])
insts['affiliation_id']=insts['affiliation_id'].apply(lambda x: x.split("I")[-1])

print(insts.columns)
insts=insts[['ror_id','affiliation_id']]

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 81
# !!   },
# !!   "id": "2620c598",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285245712,
# !!     "user_tz": 180,
# !!     "elapsed": 14,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "c5fe21df-5f05-4946-efc3-681569c8013b"
# !! }}
insts.sample()

# %%
# !! {"metadata":{
# !!   "id": "e34ffe83",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285245713,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror_to_join = ror[['ror_id','name','aliases','acronyms','labels','country','types',
                   'address','alias_len','acronyms_len','labels_len','address_len']].copy()

# %%
# !! {"metadata":{
# !!   "id": "1cb918a9",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285245998,
# !!     "user_tz": 180,
# !!     "elapsed": 295,
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
# !!   "id": "9b6e7555",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285246306,
# !!     "user_tz": 180,
# !!     "elapsed": 312,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror_to_join['country_code'] = ror_to_join['country'].apply(lambda x: x['country_code'])
ror_to_join['country_name'] = ror_to_join['country'].apply(lambda x: x['country_name'])
ror_to_join['city'] = ror_to_join['address'].apply(lambda x: x['city'])
ror_to_join['state'] = ror_to_join['address'].apply(lambda x: x['state'])
ror_to_join['region'] = ror_to_join['address'].apply(get_geoname_admin)
ror_to_join['institution'] = ror_to_join['name']

# %%
# !! {"metadata":{
# !!   "id": "b2a0125e"
# !! }}
"""
##### Looking at ROR Labels
"""

# %%
# !! {"metadata":{
# !!   "id": "9c84e085",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285246307,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
codes_to_ignore = ['ja','fa','hi','ko','bn','zh','ml','ru','el','kn','gu','mk','ne','te','hy',
                   'km','ti','kk','th','my','uk','pa','bg','ur','vi','ar','sr','he','ta','ka',
                   'am','mr','lo','mn','be','or','ba','si','ky','uz']

# %%
# !! {"metadata":{
# !!   "id": "530c77b3",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285246307,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
labels = ror_to_join['labels'].explode().dropna().reset_index()
labels['label'] = labels['labels'].apply(lambda x: x['label'])
labels['iso639'] = labels['labels'].apply(lambda x: x['iso639'])

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 745
# !!   },
# !!   "id": "47bbf00e",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285246308,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "001986d2-d9bc-4f08-ddd5-e77c8a882873"
# !! }}
labels[~labels['iso639'].isin(codes_to_ignore)].sample(20)

# %%
# !! {"metadata":{
# !!   "id": "5235bc76"
# !! }}
"""
#### Getting string beginnings
"""

# %%
# !! {"metadata":{
# !!   "id": "9719eb96"
# !! }}
"""
Looking to introduce more variety into the artificial strings so that they include some header information such as "College of .." or "Department of...".
"""

# %%
# !! {"metadata":{
# !!   "id": "50816e1c",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285247402,
# !!     "user_tz": 180,
# !!     "elapsed": 1100,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "7d94e58c-9b15-4e99-c369-3e899ffe9b7e"
# !! }}
file_path = os.path.join(base_path, 'V2/001_Exploration/ror_string_beginnings/Company')
with open(file_path, 'r') as f:
    company_begs = f.readlines()

company_begs = list(set([x.rstrip('\n') for x in company_begs]))
print(company_begs)

# %%
# !! {"metadata":{
# !!   "id": "df236ab3",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285247829,
# !!     "user_tz": 180,
# !!     "elapsed": 429,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "outputId": "c1ab7861-c5ce-49b6-8889-6094b0899bd3"
# !! }}
file_path = os.path.join(base_path, 'V2/001_Exploration/ror_string_beginnings/Education_dept')
with open(file_path, 'r') as f:
    education_dept_begs = f.readlines()

education_dept_begs = list(set([x.rstrip('\n') for x in education_dept_begs]))
print(education_dept_begs)

# %%
# !! {"metadata":{
# !!   "id": "a72aaf9a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285247830,
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
# !!   "outputId": "069c1150-b05c-4840-9663-56786483303c"
# !! }}
file_path = os.path.join(base_path, 'V2/001_Exploration/ror_string_beginnings/Education_college')
with open(file_path, 'r') as f:
    education_col_begs = f.readlines()

education_col_begs = list(set([x.rstrip('\n') for x in education_col_begs]))
print(education_col_begs)

# %%
# !! {"metadata":{
# !!   "id": "86ea4260",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248097,
# !!     "user_tz": 180,
# !!     "elapsed": 271,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "6dc711dd-5ea7-4504-db29-97ec52811bf0"
# !! }}
file_path = os.path.join(base_path, 'V2/001_Exploration/ror_string_beginnings/Education_school')
with open(file_path, 'r') as f:
    education_school_begs = f.readlines()

education_school_begs = list(set([x.rstrip('\n') for x in education_school_begs]))
print(education_school_begs)

# %%
# !! {"metadata":{
# !!   "id": "9cd5fb96",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248389,
# !!     "user_tz": 180,
# !!     "elapsed": 294,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
file_path = os.path.join(base_path, 'V2/001_Exploration/ror_string_beginnings/Healthcare')
with open(file_path, 'r') as f:
    healthcare_begs = f.readlines()

healthcare_begs = list(set([x.rstrip('\n') for x in healthcare_begs]))

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "916341c8",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248389,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "9f8dbd89-806f-41c0-e7c6-cd39384929b7"
# !! }}
all_education = []
for beg in education_col_begs:
    all_education.append(f"College of {beg}")

for beg in education_dept_begs:
    all_education.append(f"Department of {beg}")
    all_education.append(f"Dep of {beg}")
    all_education.append(f"Dept of {beg}")
    all_education.append(f"Dep. of {beg}")
    all_education.append(f"Dept. of {beg}")

for beg in education_school_begs:
    all_education.append(f"School of {beg}")
    all_education.append(f"Sch. of {beg}")
    all_education.append(f"Sch of {beg}")
len(all_education)

# %%
# !! {"metadata":{
# !!   "id": "4856f7dc",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248389,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
all_company = company_begs

# %%
# !! {"metadata":{
# !!   "id": "ffa4997e",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248389,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
all_healthcare = []
for beg in healthcare_begs:
    all_healthcare.append(f"Department of {beg}")
    all_healthcare.append(f"Dep of {beg}")
    all_healthcare.append(f"Dept of {beg}")
    all_healthcare.append(f"Dep. of {beg}")
    all_healthcare.append(f"Dept. of {beg}")

# %%
# !! {"metadata":{
# !!   "id": "ea584707"
# !! }}
"""
##### Creating the artificial affiliation strings
"""

# %%
# !! {"metadata":{
# !!   "id": "5dd28354"
# !! }}
"""
We would like to take advantage of the ROR data and use it to supplement/augment the current affiliation string data in OpenAlex. This could potentially allow for the Institution Tagger model to predict on institutions that have not yet had affiliation strings added to OpenAlex.
"""

# %%
# !! {"metadata":{
# !!   "id": "42fb9e1a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248390,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
type_preinst_dict = {'Company': all_company,
                     'Education': all_education,
                     'Healthcare': all_healthcare}

# %%
# !! {"metadata":{
# !!   "id": "1b3cc139",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248677,
# !!     "user_tz": 180,
# !!     "elapsed": 294,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def create_affiliation_string_from_scratch(institution, city, state, country, region, aliases, labels,
                                           acronyms, inst_type):
    aff_strings = []
    if aliases:
        aliases = [institution] + aliases
    else:
        aliases = [institution]
    if labels:
        for label in labels:
            if label['iso639'] in codes_to_ignore:
                pass
            else:
                aliases.append(label['label'])
    if acronyms:
        for acronym in acronyms:
            aliases.append(acronym)
    for alias in aliases:
        if "r&d" not in alias.lower():
            alias = alias.replace(" & ", " and ")
        match_string = unidecode.unidecode(alias)
        match_string = match_string.lower().replace(" ", "")
        match_string = "".join(re.findall("[a-zA-Z0-9]+", match_string))
        alias = unidecode.unidecode(alias)
        if ((state is None) & (region != city) & (city is not None) &
            (country is not None) & (region is not None)):
            region = unidecode.unidecode(region)
            country = unidecode.unidecode(country)
            city = unidecode.unidecode(city)
            aff_strings.append([f"{alias}, {city}, {region}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {region}", match_string])
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}", match_string])
            aff_strings.append([f"{alias}, {region}", match_string])
            aff_strings.append([f"{alias} {city} {region} {country}", match_string])
            aff_strings.append([f"{alias} {city} {country}", match_string])
            aff_strings.append([f"{alias} {city} {region}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias} {city}", match_string])
            aff_strings.append([f"{alias} {region}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(5):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(2):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])

        elif (state is not None) & (city is not None) & (country is not None):
            state = unidecode.unidecode(state)
            country = unidecode.unidecode(country)
            city = unidecode.unidecode(city)
            aff_strings.append([f"{alias}, {city}, {state}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {state}", match_string])
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}", match_string])
            aff_strings.append([f"{alias}, {state}", match_string])
            aff_strings.append([f"{alias} {city} {state} {country}", match_string])
            aff_strings.append([f"{alias} {city} {country}", match_string])
            aff_strings.append([f"{alias} {city} {state}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias} {city}", match_string])
            aff_strings.append([f"{alias} {state}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(1):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
        elif (city is not None) & (country is not None):
            country = unidecode.unidecode(country)
            city = unidecode.unidecode(city)
            aff_strings.append([f"{alias}, {city}, {country}", match_string])
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}", match_string])
            aff_strings.append([f"{alias} {city} {country}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias} {city}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(1):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
        elif (country is not None):
            country = unidecode.unidecode(country)
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(4):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(1):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])

        else:
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(5):
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                else:
                    for i in range(2):
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
    return aff_strings

# %%
# !! {"metadata":{
# !!   "id": "fd247555",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285248678,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_institutions_list(institution, aliases, labels, acronyms):
    aff_strings = []
    if aliases:
        aliases = [institution] + aliases
    else:
        aliases = [institution]
    if labels:
        for label in labels:
            if label['iso639'] in codes_to_ignore:
                pass
            else:
                aliases.append(label['label'])
    return aliases

# %%
# !! {"metadata":{
# !!   "id": "67c62ee0",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285314162,
# !!     "user_tz": 180,
# !!     "elapsed": 65487,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror_to_join['affs'] = ror_to_join \
.apply(lambda x: get_institutions_list(x.institution, x.aliases,
                                       x.labels, x.acronyms), axis=1)
ror_to_join['aff_string'] = ror_to_join \
.apply(lambda x: create_affiliation_string_from_scratch(x.institution, x.city, x.state,
                                                        x.country_name, x.region,
                                                        x.aliases, x.labels, x.acronyms, x.types), axis=1)
ror_to_join['aff_string_len'] = ror_to_join['aff_string'].apply(len)
ror_to_join_final = ror_to_join.explode("aff_string").copy()

# %%
# !! {"metadata":{
# !!   "id": "73ed637b"
# !! }}
"""
##### Looking to quickly get combinations of city/region/country
"""

# %%
# !! {"metadata":{
# !!   "id": "1f5467d4",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285314162,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
art_empty_affs = ror_to_join[['city','region','country_name']].dropna().copy()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "af794595",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285314162,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "fb7056e2-5323-41a6-d2b0-ae0ce6be6508"
# !! }}
art_empty_affs.shape

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 363
# !!   },
# !!   "id": "0e1e5a5a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285314163,
# !!     "user_tz": 180,
# !!     "elapsed": 9,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "b1167267-2720-4f69-e239-e9308272a4cd"
# !! }}
art_empty_affs.sample(10)

# %%
# !! {"metadata":{
# !!   "id": "806f2f6a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285316847,
# !!     "user_tz": 180,
# !!     "elapsed": 2691,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
art_empty_affs['original_affiliation_1'] = \
    art_empty_affs.apply(lambda x: f"{x.city}, {x.country_name}", axis=1)

# %%
# !! {"metadata":{
# !!   "id": "cf3fb025",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285319557,
# !!     "user_tz": 180,
# !!     "elapsed": 2713,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
art_empty_affs['original_affiliation_2'] = \
    art_empty_affs.apply(lambda x: f"{x.city}, {x.region}, {x.country_name}", axis=1)

# %%
# !! {"metadata":{
# !!   "id": "e3c15808",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285319557,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
city_country = art_empty_affs.sample(1500).drop_duplicates()\
    ['original_affiliation_1'].to_list()

# %%
# !! {"metadata":{
# !!   "id": "efea34cb",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285319557,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
city_region_country = art_empty_affs.sample(1500).drop_duplicates()\
    ['original_affiliation_2'].to_list()

# %%
# !! {"metadata":{
# !!   "id": "9e727b98"
# !! }}
"""
Some of these string will be used to train the model that only seeing a city/region/country should be a "no prediction"
"""

# %%
# !! {"metadata":{
# !!   "id": "81fc9abe",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285319984,
# !!     "user_tz": 180,
# !!     "elapsed": 431,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
pd.DataFrame(zip(city_country+city_region_country),
             columns=['original_affiliation']) \
    .drop_duplicates(subset=['original_affiliation'])\
    .to_parquet(f"{base_path}V2/artificial_empty_affs.parquet")

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "1818d865",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285322662,
# !!     "user_tz": 180,
# !!     "elapsed": 2682,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "1aea49dc-1388-4ce6-d123-9ac12e281cdc"
# !! }}
ror_to_join_final = ror_to_join[['ror_id','aff_string']].explode("aff_string").copy()
ror_to_join_final.shape

# %%
# !! {"metadata":{
# !!   "id": "9ec78746",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285325359,
# !!     "user_tz": 180,
# !!     "elapsed": 2700,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
ror_to_join_final['original_affiliation'] = \
    ror_to_join_final['aff_string'].apply(lambda x: x[0])

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 206
# !!   },
# !!   "id": "CG1dx7Dd46as",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285325360,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "4706871e-b27e-40e9-a1e7-d8579ef4d02d"
# !! }}
ror_to_join_final.head()

# %%
# !! {"metadata":{
# !!   "id": "24a322c9"
# !! }}
"""
The rest of the artificial strings are exported so that they can be combined with the historical affiliation data to create the final training data set.
"""

# %%
# !! {"metadata":{
# !!   "id": "586eb152",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285326834,
# !!     "user_tz": 180,
# !!     "elapsed": 1482,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}

ror_to_join_final.merge(insts, how='inner',
                        on='ror_id')[['original_affiliation','affiliation_id']] \
.to_parquet(f"{base_path}V2/001_Exploration/ror_strings.parquet")

# %%
# !! {"metadata":{
# !!   "id": "Xmv2vS8bCubi",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285327802,
# !!     "user_tz": 180,
# !!     "elapsed": 971,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 206
# !!   },
# !!   "outputId": "aa13e688-690e-4c0d-facc-27f854504138"
# !! }}
ror_to_join_final.merge(insts, how='inner',
                        on='ror_id')[['original_affiliation','affiliation_id']].head()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "ueOqGvj1PNeq",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732285922295,
# !!     "user_tz": 180,
# !!     "elapsed": 2752,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "f409f212-ef24-41da-a5be-7d2d71553e86"
# !! }}
#sub_p_res = subprocess.run(['pip', 'install', 'colab-convert'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
#print(sub_p_res) #<cc-cm>
#sub_p_res = subprocess.run(['colab-convert', '001_Institutions_Exploration.ipynb', '001_Institutions_Exploration.p'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
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
# !!     "version": "3.8.12"
# !!   },
# !!   "colab": {
# !!     "provenance": []
# !!   }
# !! }}
