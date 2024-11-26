# %%
# !! {"metadata":{
# !!   "id":"cc-imports"
# !! }}

#<cc-imports>

import subprocess

# %%
# !! {"metadata":{
# !!   "id": "oCjnGCtXhGv6",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287141143,
# !!     "user_tz": 180,
# !!     "elapsed": 803,
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
# !!   "executionInfo": {
# !!     "elapsed": 27208,
# !!     "status": "ok",
# !!     "timestamp": 1732287168846,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "efc67f40",
# !!   "outputId": "35b7e78b-070e-4fe5-cd26-5bb29bdbaa05"
# !! }}
import pickle
import json
import os
import math
sub_p_res = subprocess.run(['pip', 'install', 'unidecode'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
import unidecode
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from math import ceil
from sklearn.model_selection import train_test_split

# %%
# !! {"metadata":{
# !!   "id": "TbyenAE3m_7S"
# !! }}
"""

"""

# %%
# !! {"metadata":{
# !!   "id": "47f34955",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287168847,
# !!     "user_tz": 180,
# !!     "elapsed": 6,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# HuggingFace library to train a tokenizer
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 63598,
# !!     "status": "ok",
# !!     "timestamp": 1732287232440,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "nLpOEtU_Av2A",
# !!   "outputId": "6eced64b-4d4e-4585-e0de-ba70516fe2a5"
# !! }}
from google.colab import drive
drive.mount('/content/drive')

# %%
# !! {"metadata":{
# !!   "id": "eQnQibFzim3h"
# !! }}
"""
**Cambiar path para correr desde otro lado**
"""

# %%
# !! {"metadata":{
# !!   "id": "lrLMdGNEijtT",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287232440,
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


# %%
# !! {"metadata":{
# !!   "id": "80df783e"
# !! }}
"""
### Combining the training data from 001 notebook and artificial data
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 3515,
# !!     "status": "ok",
# !!     "timestamp": 1732287235951,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "b4f54477",
# !!   "outputId": "270aa13b-2527-4ce8-807b-62e251b6a412"
# !! }}
# All training samples that have less than 50 different version of the affiliation text
# ---- Created in previous notebookaffili
#lower_than = pd.read_parquet("lower_than_50.parquet")
lower_than = pd.read_parquet(f"{base_path}V2/002_Model/lower_than_50.parquet")


# All training samples that have more than 50 different version of the affiliation text
# ---- Created in previous notebook
#more_than = pd.read_parquet("more_than_50.parquet")
more_than = pd.read_parquet(f'{base_path}/V2/002_Model/more_than_50.parquet')

print(lower_than.shape)
print(more_than.shape)

# %%
# !! {"metadata":{
# !!   "id": "8f756a12",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287235952,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
full_affs_data = pd.concat([more_than, lower_than],
                           axis=0).reset_index(drop=True)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 10,
# !!     "status": "ok",
# !!     "timestamp": 1732287235952,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "EWe7_RNSLI-c",
# !!   "outputId": "c9692cbd-3a7f-480d-e612-8ab9504756bd"
# !! }}
full_affs_data.info()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 206
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 8,
# !!     "status": "ok",
# !!     "timestamp": 1732287235952,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "9sxIVVh2LO47",
# !!   "outputId": "77a29c74-a5f3-42d9-a7fd-ef6b7202f93c"
# !! }}
full_affs_data.head()

# %%
# !! {"metadata":{
# !!   "id": "fb390e26",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237188,
# !!     "user_tz": 180,
# !!     "elapsed": 1242,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
##guardamos como parquet
full_affs_data.to_parquet(f'{base_path}V2/002_Model/full_affs_data.parquet')

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 19,
# !!     "status": "ok",
# !!     "timestamp": 1732287237189,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "1035dc08",
# !!   "outputId": "8f1f9e43-7c72-48d0-b90c-973cd432ac27"
# !! }}
full_affs_data.shape

# %%
# !! {"metadata":{
# !!   "id": "dc06758d",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237189,
# !!     "user_tz": 180,
# !!     "elapsed": 18,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
full_affs_data['text_len'] = full_affs_data['original_affiliation'].apply(len)

# %%
# !! {"metadata":{
# !!   "id": "6d923ca1",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237189,
# !!     "user_tz": 180,
# !!     "elapsed": 17,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
full_affs_data = full_affs_data[full_affs_data['text_len'] < 500][['original_affiliation','affiliation_id']].copy()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 18,
# !!     "status": "ok",
# !!     "timestamp": 1732287237190,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "cccad5f6",
# !!   "outputId": "697ef122-0c4b-465f-a4f4-7d8052eafb3d"
# !! }}
full_affs_data.shape

# %%
# !! {"metadata":{
# !!   "id": "7ca1aae8",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237190,
# !!     "user_tz": 180,
# !!     "elapsed": 16,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
full_affs_data['affiliation_id'] = full_affs_data['affiliation_id'].astype('str')

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 143
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 15,
# !!     "status": "ok",
# !!     "timestamp": 1732287237190,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "SziG4V84FuXp",
# !!   "outputId": "25770371-ce0b-402c-f63b-973f8e09a5c6"
# !! }}
full_affs_data.head(n=3)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 13,
# !!     "status": "ok",
# !!     "timestamp": 1732287237190,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "Az308qsmLh9a",
# !!   "outputId": "eb8dc3d8-21bc-47f2-9a41-d7ba0deea422"
# !! }}
full_affs_data.info()

# %%
# !! {"metadata":{
# !!   "id": "9Y7Lnup3KBt2",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237190,
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
# !!   "id": "7e12f21c"
# !! }}
"""
### Processing and splitting the data
"""

# %%
# !! {"metadata":{
# !!   "id": "237023d0",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237191,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
##nueva función para preprocesar con todo el dataset

import re # Import the 're' module for regular expressions

def preprocess_affiliation_text(text):
    # Normalización de caracteres
    text = unidecode.unidecode(text)
    # Convertir a minúsculas
    text = text.lower()
    # Remover caracteres especiales (si es necesario)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Aplicar el preprocesamiento a los datos de entrenamiento, validación y prueba
full_affs_data['processed_text'] = full_affs_data['original_affiliation'].apply(preprocess_affiliation_text)

##antes estaba la sig linea
##full_affs_data['processed_text'] = full_affs_data['original_affiliation'].apply(unidecode.unidecode)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 206
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 797,
# !!     "status": "ok",
# !!     "timestamp": 1732287237979,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "011RvKgcLt30",
# !!   "outputId": "cc9e5592-f5c9-4010-da23-b03aec436fae"
# !! }}
full_affs_data.head()

# %%
# !! {"metadata":{
# !!   "id": "22e8bbfd",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287237980,
# !!     "user_tz": 180,
# !!     "elapsed": 14,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
train_data, val_data = train_test_split(full_affs_data, train_size=0.80, random_state=1)
train_data = train_data.reset_index(drop=True).copy()
val_data = val_data.reset_index(drop=True).copy()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 14,
# !!     "status": "ok",
# !!     "timestamp": 1732287237980,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "3f94e57e",
# !!   "outputId": "4a72ea28-18f8-45c5-f017-6b793cf79bf6"
# !! }}
train_data.shape

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 12,
# !!     "status": "ok",
# !!     "timestamp": 1732287237980,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "81281ab0",
# !!   "outputId": "5b9b4890-c54f-44be-d18f-7984a1d82ee1"
# !! }}
val_data.shape

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 10,
# !!     "status": "ok",
# !!     "timestamp": 1732287237980,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "1ee2242a",
# !!   "outputId": "50aaf728-6b15-4a9e-ad0c-20ab196248e8"
# !! }}
affs_list_train = train_data['processed_text'].tolist()
affs_list_val = val_data['processed_text'].tolist()
affs_list_val[:5]

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 8,
# !!     "status": "ok",
# !!     "timestamp": 1732287237980,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "ee7ec70c",
# !!   "outputId": "39d43cbb-d41b-442b-94e3-db7e2b5d0f8f"
# !! }}
try:
    os.system("rm {base_path}V2/002_Model/aff_text.txt")
    print("Done")
except:
    pass

# %%
# !! {"metadata":{
# !!   "id": "83afeec6",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287238627,
# !!     "user_tz": 180,
# !!     "elapsed": 653,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# save the affiliation text that will be used to train a tokenizer
#with open("aff_text.txt", "w") as f:
base_path = '/content/drive/MyDrive/openalex-institution-parsing/'
with open(f"{base_path}V2/002_Model/aff_text.txt", "w") as f: # Added 'w' to open in write mode
    for aff in affs_list_train:
        f.write(f"{aff}\n")

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 3,
# !!     "status": "ok",
# !!     "timestamp": 1732287238627,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "331b14ae",
# !!   "outputId": "47224bfd-d0af-42d4-9b52-8f6d0ce21ae2"
# !! }}
try:
    os.system("rm {base_path}V2/002_Model/basic_model_tokenizer")
    print("Done")
except:
    pass

# %%
# !! {"metadata":{
# !!   "id": "396120a8",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287239285,
# !!     "user_tz": 180,
# !!     "elapsed": 660,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
#full_affs_data[['processed_text','affiliation_id']].to_parquet("full_affs_data_processed.parquet")
full_affs_data[['processed_text','affiliation_id']].to_parquet(f"{base_path}V2/002_Model/full_affs_data_processed.parquet")


# %%
# !! {"metadata":{
# !!   "id": "0a90e2b7"
# !! }}
"""
### Creating the tokenizer for the basic model
"""

# %%
# !! {"metadata":{
# !!   "id": "724890a7",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287240657,
# !!     "user_tz": 180,
# !!     "elapsed": 1374,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
wordpiece_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# NFD Unicode, lowercase, and getting rid of accents (to make sure text is as readable as possible)
wordpiece_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# Splitting on whitespace
wordpiece_tokenizer.pre_tokenizer = Whitespace()

# Training a tokenizer on the training dataset
trainer = WordPieceTrainer(vocab_size=3816, special_tokens=["[UNK]"])
#files = ["aff_text.txt"]
files = [f"{base_path}V2/002_Model/aff_text.txt"]

wordpiece_tokenizer.train(files, trainer)

#wordpiece_tokenizer.save("basic_model_tokenizer")
wordpiece_tokenizer.save(f"{base_path}V2/002_Model/basic_model_tokenizer")

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 4,
# !!     "status": "ok",
# !!     "timestamp": 1732287240657,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "t6ebUlIPG1Q-",
# !!   "outputId": "2a0afc8e-70f1-41a6-8219-4647f845103f"
# !! }}
print(wordpiece_tokenizer)

# %%
# !! {"metadata":{
# !!   "id": "c37e4b4b"
# !! }}
"""
### Further processing of data with tokenizer
"""

# %%
# !! {"metadata":{
# !!   "id": "1a923dfb",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287240657,
# !!     "user_tz": 180,
# !!     "elapsed": 3,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def max_len_and_pad(tok_sent):
    """
    Truncates sequences with length higher than max_len and also pads the sequence
    with zeroes up to the max_len.
    """
    max_len = 128
    tok_sent = tok_sent[:max_len]
    tok_sent = tok_sent + [0]*(max_len - len(tok_sent))
    return tok_sent

def create_affiliation_vocab(x):
    """
    Checks if affiliation is in vocab and if not, adds to the vocab.
    """
    if x not in affiliation_vocab.keys():
        affiliation_vocab[x]=len(affiliation_vocab)
    return [affiliation_vocab[x]]




# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 115
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 1423,
# !!     "status": "ok",
# !!     "timestamp": 1732287242078,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "deb6cb01",
# !!   "outputId": "557c40b1-bf3a-4fbd-fb4e-dc712de9a318"
# !! }}
# initializing an empty affiliation vocab
affiliation_vocab = {}

# tokenizing the training dataset
tokenized_output = []
for i in affs_list_train:
    tokenized_output.append(wordpiece_tokenizer.encode(i).ids)

train_data['original_affiliation_tok'] = tokenized_output
train_data['original_affiliation_tok'].head(1)

# %%
# !! {"metadata":{
# !!   "id": "37d80c41",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287242079,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# tokenizing the validation dataset
tokenized_output = []
for i in affs_list_val:
    tokenized_output.append(wordpiece_tokenizer.encode(i).ids)

val_data['original_affiliation_tok'] = tokenized_output

# %%
# !! {"metadata":{
# !!   "id": "f840adc2",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287242079,
# !!     "user_tz": 180,
# !!     "elapsed": 3,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# applying max length cutoff and padding
train_data['original_affiliation_model_input'] = train_data['original_affiliation_tok'].apply(max_len_and_pad)
val_data['original_affiliation_model_input'] = val_data['original_affiliation_tok'].apply(max_len_and_pad)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 115
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 499,
# !!     "status": "ok",
# !!     "timestamp": 1732287242574,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "yeROrBcQIKqs",
# !!   "outputId": "f519100d-19e5-4c57-a655-7509f7788861"
# !! }}
train_data['original_affiliation_tok'].apply(max_len_and_pad).head(1)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 115
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 29,
# !!     "status": "ok",
# !!     "timestamp": 1732287242575,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "Yg922Nv5I3fG",
# !!   "outputId": "15271f63-bde1-4cb6-c63f-6d5aeec8f99c"
# !! }}
val_data['original_affiliation_tok'].apply(max_len_and_pad).head(1)

# %%
# !! {"metadata":{
# !!   "id": "331f34e9",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287242576,
# !!     "user_tz": 180,
# !!     "elapsed": 29,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# creating the label affiliation vocab
train_data['label'] = train_data['affiliation_id'].apply(lambda x: create_affiliation_vocab(x))

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 178
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 28,
# !!     "status": "ok",
# !!     "timestamp": 1732287242576,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "y2gVhT6SJovo",
# !!   "outputId": "d6c3fc4b-e336-4337-f021-183f119228c0"
# !! }}
train_data['affiliation_id'].apply(lambda x: create_affiliation_vocab(x)).head(3)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 27,
# !!     "status": "ok",
# !!     "timestamp": 1732287242576,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "Klemd5SvJ_lD",
# !!   "outputId": "bd773669-49db-4564-e6fd-b66091e73e09"
# !! }}
train_data.shape

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 24,
# !!     "status": "ok",
# !!     "timestamp": 1732287242576,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "1f9e15f6",
# !!   "outputId": "338a2fcd-fa21-49b0-b1df-cfb90d0d9e66"
# !! }}
len(affiliation_vocab)

# %%
# !! {"metadata":{
# !!   "id": "afdf7620",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287242576,
# !!     "user_tz": 180,
# !!     "elapsed": 22,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
val_data['label'] = val_data['affiliation_id'].apply(lambda x: [affiliation_vocab.get(x)])

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 178
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 21,
# !!     "status": "ok",
# !!     "timestamp": 1732287242576,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "64D2MOscJsju",
# !!   "outputId": "131bc842-2ed2-461b-a615-b0814c7c20ac"
# !! }}
val_data['affiliation_id'].apply(lambda x: [affiliation_vocab.get(x)]).head(3)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 21,
# !!     "status": "ok",
# !!     "timestamp": 1732287242577,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "wd6yKig7J6c9",
# !!   "outputId": "91de7584-b200-4d99-cb2d-013781f91375"
# !! }}
val_data.shape

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 19,
# !!     "status": "ok",
# !!     "timestamp": 1732287242577,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "eGLXNwWyqTcI",
# !!   "outputId": "b56a8d13-1a4f-4b72-a525-da6b9675df09"
# !! }}
dict(list(affiliation_vocab.items())[0:10])

# %%
# !! {"metadata":{
# !!   "id": "t_SJ7MuOdqfW",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287242577,
# !!     "user_tz": 180,
# !!     "elapsed": 15,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}



# %%
# !! {"metadata":{
# !!   "id": "bee93ca0",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287246031,
# !!     "user_tz": 180,
# !!     "elapsed": 3469,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
#train_data.to_parquet("train_data.parquet")
#val_data.to_parquet("val_data.parquet")

train_data.to_parquet(f"{base_path}V2/002_Model/training_data/train_data.parquet")
val_data.to_parquet(f"{base_path}V2/002_Model/training_data/val_data.parquet")

# %%
# !! {"metadata":{
# !!   "id": "d6994186",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248001,
# !!     "user_tz": 180,
# !!     "elapsed": 1973,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# saving the affiliation vocab
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","wb") as f:
    pickle.dump(affiliation_vocab, f)


with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","rb") as f:
    affiliation_vocab = pickle.load(f)


###agrego "-1" para evitar futuros errores
affiliation_vocab[-1]=len(affiliation_vocab)
affiliation_vocab

with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","wb") as f:
    pickle.dump(affiliation_vocab, f)


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 15,
# !!     "status": "ok",
# !!     "timestamp": 1732287248001,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "9LUEfz7MeusO",
# !!   "outputId": "0e1e7420-4e2b-4677-c318-b2562d56209c"
# !! }}
len(affiliation_vocab)

# %%
# !! {"metadata":{
# !!   "id": "b9e469e1"
# !! }}
"""
### Creating TFRecords from the training and validation datasets
"""

# %%
# !! {"metadata":{
# !!   "id": "479533d5",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248001,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
train_data = pd.read_parquet(f"{base_path}V2/002_Model/training_data/train_data.parquet")

# %%
# !! {"metadata":{
# !!   "id": "98e2dc29",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248001,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
val_data = pd.read_parquet(f"{base_path}V2/002_Model/training_data/val_data.parquet")

# %%
# !! {"metadata":{
# !!   "id": "0c094892",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248002,
# !!     "user_tz": 180,
# !!     "elapsed": 14,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# saving the affiliation vocab
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","rb") as f:
    affiliation_vocab = pickle.load(f)

# %%
# !! {"metadata":{
# !!   "id": "e56231dc",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248002,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def create_tfrecords_dataset(data, iter_num, dataset_type='train'):
    """
    Creates a TF Dataset that can then be saved to a file to make it faster to read
    data during training and allow for transferring of data between compute instances.
    """
    ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(data['original_affiliation_model_input'].to_list()),
                              tf.data.Dataset.from_tensor_slices(data['label'].to_list())))

    serialized_features_dataset = ds.map(tf_serialize_example)

    filename = f"{base_path}V2/002_Model/training_data/{dataset_type}/{str(iter_num).zfill(4)}.tfrecord"
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

# %%
# !! {"metadata":{
# !!   "id": "9c27f6c6",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248002,
# !!     "user_tz": 180,
# !!     "elapsed": 12,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def tf_serialize_example(f0, f1):
    """
    Serialization function.
    """
    tf_string = tf.py_function(serialize_example, (f0, f1), tf.string)
    return tf.reshape(tf_string, ())

# %%
# !! {"metadata":{
# !!   "id": "56124ea7",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248002,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def serialize_example(features, label):
    """
    Takes in features and outputs them to a serialized string that can be written to
    a file using the TFRecord Writer.
    """
    features_list = tf.train.Int64List(value=features.numpy().tolist())
    label_list = tf.train.Int64List(value=label.numpy().tolist())

    features_feature = tf.train.Feature(int64_list = features_list)
    label_feature = tf.train.Feature(int64_list = label_list)

    features_for_example = {
        'features': features_feature,
        'label': label_feature
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))

    return example_proto.SerializeToString()

# %%
# !! {"metadata":{
# !!   "id": "ab12d69d",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287248002,
# !!     "user_tz": 180,
# !!     "elapsed": 10,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# Making sure data is in the correct format before going into TFRecord
train_data['original_affiliation_model_input'] = train_data['original_affiliation_model_input'] \
.apply(lambda x: np.asarray(x, dtype=np.int64))

val_data['original_affiliation_model_input'] = val_data['original_affiliation_model_input'] \
.apply(lambda x: np.asarray(x, dtype=np.int64))

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 403,
# !!     "status": "ok",
# !!     "timestamp": 1732287248396,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "b99925a4",
# !!   "outputId": "10f66030-1f85-4d53-a562-3e2345540c98"
# !! }}
os.system(f"mkdir -p {base_path}V2/002_Model/training_data/train/")
os.system(f"mkdir -p {base_path}V2/002_Model/training_data/val/")
print("Done")

# %%
# !! {"metadata":{
# !!   "id": "bba1f308"
# !! }}
"""
#### Creating the Train Dataset
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 14359,
# !!     "status": "ok",
# !!     "timestamp": 1732287262753,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "7255fdbb",
# !!   "outputId": "01e54fdc-0219-4d03-bd44-626b51f68b23",
# !!   "scrolled": true
# !! }}
#<cc-ac> %%time
for i in range(ceil(train_data.shape[0]/500000)):
    print(i)
    low = i*500000
    high = (i+1)*500000
    create_tfrecords_dataset(train_data.iloc[low:high,:], i, 'train')

# %%
# !! {"metadata":{
# !!   "id": "42b9caa5"
# !! }}
"""
#### Creating the Validation Dataset
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 2802,
# !!     "status": "ok",
# !!     "timestamp": 1732287265551,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "bc8d38f6",
# !!   "outputId": "a0a09c12-52b2-40a3-b8e1-d719468c2e20"
# !! }}
#<cc-ac> %%time
for i in range(ceil(val_data.shape[0]/60000)):
    print(i)
    low = i*60000
    high = (i+1)*60000
    create_tfrecords_dataset(val_data.iloc[low:high,:], i, 'val')

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 421
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 20,
# !!     "status": "ok",
# !!     "timestamp": 1732287265552,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "qeIPehUJHy55",
# !!   "outputId": "6f1615d7-3cd0-4679-8b36-2b40d5c4a2f7"
# !! }}
train_data.head(n=8)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 19,
# !!     "status": "ok",
# !!     "timestamp": 1732287265552,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "DusJV8Y1H-1k",
# !!   "outputId": "cd28c51c-f556-4bc3-d11c-cc22e00a2bd8"
# !! }}
train_data.info()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 195
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 16,
# !!     "status": "ok",
# !!     "timestamp": 1732287265552,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "tTKZcbUbH2ub",
# !!   "outputId": "11ddebf8-3928-4430-bcf3-8f594ca78b2c"
# !! }}
val_data.head(n=3)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 14,
# !!     "status": "ok",
# !!     "timestamp": 1732287265552,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "WDkqiG4HICC1",
# !!   "outputId": "e5479134-c933-49a3-9335-ede7cf8cae5c"
# !! }}
val_data.info()

# %%
# !! {"metadata":{
# !!   "id": "PeoHZcuiH1_V",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287265552,
# !!     "user_tz": 180,
# !!     "elapsed": 12,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def get_dataset(path, data_type='train'):
    """
    Takes in a path to the TFRecords and returns a TF Dataset to be used for training.
    """
    tfrecords = [f"{path}{data_type}/{x}" for x in os.listdir(f"{path}{data_type}/") if x.endswith('tfrecord')]
    tfrecords.sort()

    raw_dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO)
    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTO)

    parsed_dataset = parsed_dataset.apply(tf.data.experimental.dense_to_ragged_batch(512,drop_remainder=True))
    return parsed_dataset

# %%
# !! {"metadata":{
# !!   "id": "S8I-J-n9H8Io",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287265552,
# !!     "user_tz": 180,
# !!     "elapsed": 11,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def _parse_function(example_proto):
    """
    Parses the TFRecord file.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature((128,), tf.int64),
        'label': tf.io.FixedLenFeature((1,), tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    features = example['features']
    label = example['label'][0]

    return features, label

# %%
# !! {"metadata":{
# !!   "id": "edc6ff48"
# !! }}
"""
### Loading the Data
"""

# %%
# !! {"metadata":{
# !!   "id": "814c8600",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287266328,
# !!     "user_tz": 180,
# !!     "elapsed": 787,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "3632d3c4-95d7-4fe1-d645-3d6c611a407e"
# !! }}
train_data_path = f"{base_path}V2/002_Model/training_data/"
AUTO = tf.data.experimental.AUTOTUNE
training_data = get_dataset(train_data_path, data_type='train')
validation_data = get_dataset(train_data_path, data_type='val')

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 272
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 19,
# !!     "status": "ok",
# !!     "timestamp": 1732287266329,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "vtAVX01uDCxX",
# !!   "outputId": "844f021c-0a21-45a2-a97d-526c930293a4"
# !! }}
val_data.isna().sum()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 272
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 17,
# !!     "status": "ok",
# !!     "timestamp": 1732287266329,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "UVXsbBwY39RD",
# !!   "outputId": "9e20c0ae-55a7-4605-8d52-cab3c07e6585"
# !! }}
train_data.isna().sum()

# %%
# !! {"metadata":{
# !!   "id": "c4oGgu8k_NLj"
# !! }}
"""

"""

# %%
# !! {"metadata":{
# !!   "id": "035b0a6a"
# !! }}
"""
### Load Vocab
"""

# %%
# !! {"metadata":{
# !!   "id": "3a907216",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287266329,
# !!     "user_tz": 180,
# !!     "elapsed": 16,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# Loading the affiliation (target) vocab
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","rb") as f:
    affiliation_vocab_id = pickle.load(f)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 16,
# !!     "status": "ok",
# !!     "timestamp": 1732287266329,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "c5d53f4f",
# !!   "outputId": "baad9114-3ee4-4dff-905e-3196182f2082"
# !! }}
affiliation_vocab_id = {int(i):int(j) for i,j in affiliation_vocab_id.items()}
print(affiliation_vocab_id)
inverse_affiliation_vocab_id = {(i):(j) for j,i in affiliation_vocab_id.items()}
print(inverse_affiliation_vocab_id)

# %%
# !! {"metadata":{
# !!   "id": "9a5ac75e"
# !! }}
"""
### Creating Model
"""

# %%
# !! {"metadata":{
# !!   "id": "950e87ee",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287266329,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# Hyperparameters to tune
emb_size = 256
max_len = 128
num_layers = 3  ## se cambió de 6 a 3
num_heads = 8
dense_1 = 2048
dense_2 = 1024
learn_rate = 0.01 ## la cambié de 0.0004 a 0.001

# %%
# !! {"metadata":{
# !!   "id": "ecbe2c8d",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287266330,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
def scheduler(epoch, curr_lr):
    """
    Setting up a exponentially decaying learning rate.
    """
    rampup_epochs = 2
    exp_decay = 0.17
    def lr(epoch, beg_lr, rampup_epochs, exp_decay):
        if epoch < rampup_epochs:
            return beg_lr
        else:
            return beg_lr * math.exp(-exp_decay * epoch)
    return lr(epoch, start_lr, rampup_epochs, exp_decay)

# %%
# !! {"metadata":{
# !!   "id": "191ab1ec",
# !!   "scrolled": true,
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287268586,
# !!     "user_tz": 180,
# !!     "elapsed": 2269,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   }
# !! }}
# Allow for use of multiple GPUs
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    # Model Inputs
    tokenized_aff_string_ids = tf.keras.layers.Input((128,), dtype=tf.int64, name='tokenized_aff_string_input')

    # Embedding Layers
    tokenized_aff_string_emb_layer = tf.keras.layers.Embedding(input_dim=3816,
                                                               output_dim=int(emb_size),
                                                               mask_zero=True,
                                                               trainable=True,
                                                               name="tokenized_aff_string_embedding")

    tokenized_aff_string_embs = tokenized_aff_string_emb_layer(tokenized_aff_string_ids)

    # First dense layer
    dense_output = tf.keras.layers.Dense(int(dense_1), activation='relu',
                                             kernel_regularizer='L2', name="dense_1")(tokenized_aff_string_embs)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_1")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")(dense_output)
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(dense_output)

    # Second dense layer
    dense_output = tf.keras.layers.Dense(int(dense_2), activation='relu',
                                             kernel_regularizer='L2', name="dense_2")(pooled_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_2")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")(dense_output)

    # Last dense layer
    final_output = tf.keras.layers.Dense(len(affiliation_vocab_id), activation='softmax', name='cls')(dense_output)

    model = tf.keras.Model(inputs=tokenized_aff_string_ids, outputs=final_output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9,
                                                     beta_2=0.99),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    curr_date = datetime.now().strftime("%Y%m%d")

    filepath_1 = f"./models/{curr_date}_{dense_1}d1_{dense_2}d2/" \


    filepath = filepath_1 + "model_epoch{epoch:02d}ckpt.keras"

    # Adding in checkpointing
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                          verbose=0, save_best_only=False,
                                                          save_weights_only=False, mode='auto',
                                                          save_freq='epoch')

    # Adding in early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)

    start_lr = float(learn_rate)

    # Adding in a learning rate schedule to decrease learning rate in later epochs
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    callbacks = [model_checkpoint, early_stopping, lr_schedule]


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 573
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 15,
# !!     "status": "ok",
# !!     "timestamp": 1732287268587,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "xoTGAWBqBgEW",
# !!   "outputId": "10658041-4075-40f9-9434-63c8a2c5d717"
# !! }}
model.summary()

# %%
# !! {"metadata":{
# !!   "id": "ada74e47"
# !! }}
"""
### Training the Model
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 72
# !!   },
# !!   "id": "sg792LHg4fW4",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287268587,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Mar\u00eda Fernanda Artola",
# !!       "userId": "10569145896499623152"
# !!     }
# !!   },
# !!   "outputId": "8f405cc3-c864-44b2-ef3e-1d9212453a51"
# !! }}
"""Ver keras tuner para grid search
from sklearn.model_selection import GridSearchCV

param_grid = dict(epochs=[5,10,20,30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring='accuracy')
grid_result = grid.fit(training_data, validation_data, callbacks=callbacks)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))"""

# %%
# !! {"metadata":{
# !!   "id": "Q4XYzra55D3d"
# !! }}
"""
The ***batch size*** in iterative gradient descent is the number of patterns shown to the network before the weights are updated. It is also an optimization in the training of the network, defining how many patterns to read at a time and keep in memory.

The ***number of epochs*** is the number of times the entire training dataset is shown to the network during training. Some networks are sensitive to the batch size, such as LSTM recurrent neural networks and Convolutional Neural Networks.
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "b613d5da",
# !!   "scrolled": true,
# !!   "outputId": "ebca3670-ab46-45a5-af55-7504fb3049c9"
# !! }}
history = model.fit(training_data, epochs=20, validation_data=validation_data, verbose=1, callbacks=callbacks)

# %%
# !! {"metadata":{
# !!   "id": "kiDm2PE8CnWM"
# !! }}
model.summary()

# %%
# !! {"metadata":{
# !!   "id": "nQ9BT0joWMOj"
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "92b07c45"
# !! }}
json.dump(str(history.history), open(f"{filepath_1}_25EPOCHS_HISTORY.json", 'w+'))

model.save(f'{base_path}V2/002_Model/Result_basic_model/modelo_basico0411.keras')
# %%
print('FINALIZADO OK')

# %%
# !! {"metadata":{
# !!   "id": "x97EjUIDGd4X"
# !! }}
model.export(f'{base_path}V2/002_Model/Result_basic_model/modelo_basico_export')


# %%
# !! {"metadata":{
# !!   "id": "ApCtFykFeEbq"
# !! }}
# para guardar archivo del modelo (codigo original no genera los archivos que se levantan en la siguiente notebook)

import os
import tensorflow as tf

# Definir el directorio en la carpeta actual donde deseas guardar el modelo
saved_model_dir = f'{base_path}V2/002_Model/Result_basic_model/basic_model/'

# Crear el directorio si no existe
os.makedirs(saved_model_dir, exist_ok=True)

# Guardar el modelo en formato SavedModel usando tf.saved_model.save()
tf.saved_model.save(model, saved_model_dir)
print(f"Modelo guardado en {saved_model_dir}")

# %%
# !! {"metadata":{
# !!   "id": "24pwoKx9-YbC"
# !! }}
## opciones para guardar el modelo
##https://medium.com/swlh/saving-and-loading-of-keras-sequential-and-functional-models-73ce704561f4

# %%
# !! {"metadata":{
# !!   "id": "DhgOxxLe-TdC"
# !! }}
ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
ax.grid()
_ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
_ = ax.legend(["Training loss", "Trainig accuracy"])

# %%
# !! {"metadata":{
# !!   "id": "nLrfOX2x-ebT"
# !! }}
#https://www.architecture-performance.fr/ap_blog/saving-a-tf-keras-model-with-data-normalization/

# %%
# !! {"metadata":{
# !!   "id": "gtklZA0UU6G_"
# !! }}
training_data

# %%
# !! {"metadata":{
# !!   "id": "qNRSqJXqeIIC"
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "kea-WhOWUSmr"
# !! }}
###########################
### Probando el predict ###
###########################

import tensorflow as tf
import pandas as pd
import numpy as np

# Extract the relevant data
a = val_data.iloc[1:10, 4].tolist()
basic_tok_tensor = tf.convert_to_tensor(a, dtype=tf.int64)
# Make predictions
tf.math.top_k(model.predict(basic_tok_tensor),2)



# %%
# !! {"metadata":{
# !!   "id": "BwRzA3Xio_gS"
# !! }}
#Lo que da la label es la posición que tiene determinada institución en el diccionario.

# %%
# !! {"metadata":{
# !!   "id": "k7DJgIoyqUQG"
# !! }}
### Opción 3: con dataset creado para argentina (Santiago)
all_data = pd.read_csv(f'{base_path}V2/testeo_ar.tsv',sep="\t")
all_data['labels'] = all_data['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])
all_data=all_data.explode('labels')
all_data['labels'] = all_data['labels'].astype(int)
print(all_data.info())
all_data.head(3)
all_data.shape

# %%
# !! {"metadata":{
# !!   "id": "xEHWtab_XHG2"
# !! }}
import unidecode
to_predict=all_data
# Decode text so only ASCII characters are used

import re # Import the 're' module for regular expressions
import pandas as pd

def preprocess_affiliation_text(text):
    # Normalización de caracteres
    # Call the unidecode function within the unidecode module
    text = unidecode.unidecode(text)
    # Convertir a minúsculas
    text = text.lower()
    # Remover caracteres especiales (si es necesario)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Aplicar el preprocesamiento a los datos de entrenamiento, validación y prueba
to_predict['decoded_text'] = to_predict['affiliation_string'] .apply(preprocess_affiliation_text)
print(to_predict.head(2))

decoded_text=to_predict
decoded_text=pd.DataFrame(decoded_text)
print(decoded_text.head(2))

#basic_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, "basic_model_tokenizer"))
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
model_path1 = f'{base_path}V2/002_Model/'
basic_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path1, "basic_model_tokenizer"))
###
decoded_text['basic_tok_data'] = basic_tokenizer(to_predict['decoded_text'].tolist())['input_ids']
###
decoded_text['basic_tok_data'] = [max_len_and_pad(x) for x in decoded_text['basic_tok_data']]
print(decoded_text['basic_tok_data'].shape)
print(decoded_text.head(2))

# Convert the padded data to a TensorFlow tensor
basic_tok_data=decoded_text['basic_tok_data'].tolist()
basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)

# Proceed with the rest of your code
all_scores, all_labels = tf.math.top_k(model.predict(basic_tok_tensor),1)  # Use the padded tensor here
all_scores = all_scores.numpy().tolist()
all_labels = all_labels.numpy().tolist()

# %%
# !! {"metadata":{
# !!   "id": "kEHogU3wdPUm"
# !! }}
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
# Import TFSMLayer from tensorflow.keras.layers
from tensorflow.keras.layers import TFSMLayer

basic_model_path=f'{base_path}V2/002_Model/Result_basic_model/basic_model/'
# Instead of using tf.keras.models.load_model, use TFSMLayer to load the SavedModel
basic_model = TFSMLayer(basic_model_path, call_endpoint='serving_default')
#basic_model = tf.keras.models.load_model(os.path.join(model_path, "basic_model"), compile=False)
#basic_model.trainable = False

# Convert the padded data to a TensorFlow tensor
basic_tok_data=decoded_text['basic_tok_data'].tolist()
basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)

# Cast the tensor to float32
#basic_tok_tensor = tf.cast(basic_tok_tensor, dtype=tf.float32)

# Proceed with the rest of your code
# Get the prediction output from the model
predictions = basic_model(basic_tok_tensor)

# Extract the 'output_0' tensor from the predictions dictionary
model_output = predictions['output_0']

# Apply tf.math.top_k to the extracted tensor
all_scores, all_labels = tf.math.top_k(model_output, 3)  # Use the extracted tensor here
all_scores = all_scores.numpy().tolist()
all_labels = all_labels.numpy().tolist()

print(all_scores)
print(all_labels)

# %%
# !! {"metadata":{
# !!   "id": "AAvubODXWoHu"
# !! }}
sub_p_res = subprocess.run(['pip', 'install', 'colab-convert'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['colab-convert', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/002a_basic_model_0411.ipynb', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/002a_basic_model_0411.p'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>


# %%
# !! {"main_metadata":{
# !!   "accelerator": "GPU",
# !!   "colab": {
# !!     "gpuType": "T4",
# !!     "provenance": []
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
# !!   }
# !! }}
