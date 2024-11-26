# %%
# !! {"metadata":{
# !!   "id":"cc-imports"
# !! }}

#<cc-imports>

import subprocess

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 57136,
# !!     "status": "ok",
# !!     "timestamp": 1732287745935,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "__Wd2rZyFvLW",
# !!   "outputId": "3a0a2f27-421d-4601-cd50-7f92b00f18c3"
# !! }}
sub_p_res = subprocess.run(['pip', 'install', '--upgrade', 'transformers', '##', 'Upgraded', 'to', '4.39.2', 'version'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'tensorflow', '##==2.11.0'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'tf-keras'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
# importing os module
import os

sub_p_res = subprocess.run(['pip', 'install', '--upgrade', 'pip'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', '--upgrade', 'pyarrow'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>

sub_p_res = subprocess.run(['pip', 'install', 'datasets'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'transformers', 'tensorflow', 'datasets'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>


# Get the list of user's
env_var = os.environ
os.environ['TF_USE_LEGACY_KERAS'] = '1' ### I’ve set up the legacy version

# %%
# !! {"metadata":{
# !!   "id": "efc67f40",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287748665,
# !!     "user_tz": 180,
# !!     "elapsed": 2734,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
import pickle
import json
import pandas as pd
pd.set_option("display.max_colwidth", None)
import numpy as np

from collections import Counter
from math import ceil
from sklearn.model_selection import train_test_split

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 30982,
# !!     "status": "ok",
# !!     "timestamp": 1732287779645,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "9b1745b7",
# !!   "outputId": "bfe66c27-b0fb-42d9-a6b7-39753ffabaff"
# !! }}
import tensorflow as tf
sub_p_res = subprocess.run(['pip', 'install', 'datasets'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from datasets import load_dataset
from transformers import create_optimizer, TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, TFDistilBertForSequenceClassification
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# %%
# !! {"metadata":{
# !!   "id": "fb8636a8"
# !! }}
"""
### Loading Affiliation Dictionary
"""

# %%
# !! {"metadata":{
# !!   "id": "hdkcvpMyZjJL",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287803411,
# !!     "user_tz": 180,
# !!     "elapsed": 23770,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "c8d88a94-ce70-41b4-cf28-ddc158407357"
# !! }}
from google.colab import drive
drive.mount('/content/drive')

# %%
# !! {"metadata":{
# !!   "id": "F9whHRndzWBh"
# !! }}
"""
**Cambiar path**
"""

# %%
# !! {"metadata":{
# !!   "id": "eiL0l9ddzRgX",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287803412,
# !!     "user_tz": 180,
# !!     "elapsed": 18,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
import os
base_path = '/content/drive/MyDrive/openalex-institution-parsing/'


# %%
# !! {"metadata":{
# !!   "id": "78df9497",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287803412,
# !!     "user_tz": 180,
# !!     "elapsed": 17,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "b58d592a-5e42-4db0-bd49-bda73b13601c"
# !! }}
# Loading the affiliation (target) vocab
import pickle
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","rb") as f:
  affiliation_vocab = pickle.load(f)

affiliation_vocab = {int(i):int(j) for i,j in affiliation_vocab.items()}
inverse_affiliation_vocab = {i:j for j,i in affiliation_vocab.items()}
print("inverse_affiliation_vocab")
print(inverse_affiliation_vocab)

# %%
# !! {"metadata":{
# !!   "id": "75f24a8f",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287803413,
# !!     "user_tz": 180,
# !!     "elapsed": 16,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","wb") as f:
    pickle.dump(affiliation_vocab, f)

# %%
# !! {"metadata":{
# !!   "id": "0c0e69c0",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287803413,
# !!     "user_tz": 180,
# !!     "elapsed": 15,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "41e3cc51-fd16-4acf-ee96-60d9f5816de1"
# !! }}
len(affiliation_vocab)

# %%
# !! {"metadata":{
# !!   "id": "a0e8e174"
# !! }}
"""
### Tokenizing Affiliation String
"""

# %%
# !! {"metadata":{
# !!   "id": "0f6cd658",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 274,
# !!     "referenced_widgets": [
# !!       "6c231dc7b1ff449f87b1486da2591276",
# !!       "4bc2ffd625314bf18a84c96e9f515e9e",
# !!       "ed57b06e330c47638b6436674ffaa16e",
# !!       "49e1e838a66945c0894816c17e8d86d3",
# !!       "c7268ca62435481cb3ab821611be3f1e",
# !!       "1e26d73c6ad54e0b91868f39816d4a3b",
# !!       "653f505efad54d67a5e8f3b2f2b4ac2e",
# !!       "4060fa244c70445f8a4f88a8a6f21919",
# !!       "6874fb7198324e2d9b421c50b6675035",
# !!       "90d24853607143c1982652abfe2fcb9b",
# !!       "7c53e5a289fa47d983f2d1b4c08a5501",
# !!       "c89531dd3eec4d978dc9eb7fd34d08e3",
# !!       "2943023c02d44014bdc2c0ae7969fc04",
# !!       "5cb4f065030e47708dcf98a89fae95bc",
# !!       "c8f8b0950a98473da5c62e67d372acf8",
# !!       "ee95e5b85e394395abc307a5e060c21b",
# !!       "a8d998e0105f47a6aa0a653365ad5139",
# !!       "87272c0ea53145b99e59f465da032708",
# !!       "cca92b3179604d20a9fbcd0bb0e3755e",
# !!       "b220022e8f204e27a12105f5dbb52d60",
# !!       "468b61c5bb084a9294c600d911b3d7be",
# !!       "f300d22a64ef48aea1b22a5d071b522c",
# !!       "93dd09198cb443e3ad1884632ebac01b",
# !!       "5fd6701b6e2b49c89879f2b88125343a",
# !!       "205b984723c34b3da8e69f5318c443de",
# !!       "28847f4bb0c44fa0837b5ebd58df5c01",
# !!       "0c5e697fb80e4619808beaf0fe56f6cb",
# !!       "f0b5354c481c4494a572d027cea8e874",
# !!       "70ade11ba70a40c08418cb8c89154c19",
# !!       "a0d253bf7d3245639bb3523f64c8c8b1",
# !!       "8e3af37de70b4f1fb44225065c25076f",
# !!       "f529487d6f0b454b854f9fd550796f2b",
# !!       "5b8212d3d80f432b8cbfe9bb87f2e45b",
# !!       "4d91076cb18245b09f146b0099da5563",
# !!       "c5c909db859c41a59c883c077f9c3998",
# !!       "b2b4d86760614ca2aeca5439c8f3f50d",
# !!       "225b65f82ccd4a92a0d36b24ce9d5130",
# !!       "871b7cb82d1140c1b0e7c96c93ba40bf",
# !!       "7b8936db9f284c648802fff2f0448432",
# !!       "b32f0110a9214e7cb3c34af64356f754",
# !!       "da2ab0cfb7b14787bcb659f456213cb5",
# !!       "12938abc05a74e359064114a0896607c",
# !!       "7de0dfd525364ed19c752677ecdddc89",
# !!       "2c7428c22d2e464c928d77e443b9d55a"
# !!     ]
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287805752,
# !!     "user_tz": 180,
# !!     "elapsed": 2348,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "d3897f51-e752-4533-c248-9a354fabbd92"
# !! }}
# Loading the standard DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", return_tensors='tf')

# %%
# !! {"metadata":{
# !!   "id": "rqYGKPg4Kkm8",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287805752,
# !!     "user_tz": 180,
# !!     "elapsed": 6,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "38e2337e-6187-4acd-c1db-fde474d7d55f"
# !! }}
tokenizer

# %%
# !! {"metadata":{
# !!   "id": "4ef5701a",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 81,
# !!     "referenced_widgets": [
# !!       "e8dda68557e9440d8209cc0bf3ff8c64",
# !!       "9202a20b880648259844aafce6f7e020",
# !!       "7aee38a33a1f4ae297f63128e4bc9650",
# !!       "5dbe7f1255484136be03d3a4f051b691",
# !!       "15f1fb12ff56436cae5b1e239556d11f",
# !!       "f480f34659de4023b8f97b9433e65695",
# !!       "3172052759934d6ab248500bbb373946",
# !!       "2da6f945f6dc46e68ab86ffbb7ba8902",
# !!       "a05cc81428744913a2886051b64ba3d6",
# !!       "c88a4137f3e043f5a7c98da1f99968ba",
# !!       "d2d393baff4b40a4b49294fd3e7a3af4",
# !!       "0c67b99e3b5b4b87b54d91f0c72c3685",
# !!       "f95227c3f3674b4c9343b61a38a132f3",
# !!       "936fae792f2b45b38430b24182728429",
# !!       "3fae53959f604744bde9cd11957b81ce",
# !!       "4faa798380b5419584cebc97b1dcc57d",
# !!       "e5e8a9f17dfa4d1eaf00560f57e369fa",
# !!       "072c35420af94b5ba08ee675cd20122f",
# !!       "d93f566d4bf6487fb4ef56eaf4b489d1",
# !!       "246f621978d34c60adae1d2c1dbb84a2",
# !!       "1e44b863cfd14b14af98548199b2ceb2",
# !!       "5428080a263f4947aa0419558a106f97"
# !!     ]
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287809989,
# !!     "user_tz": 180,
# !!     "elapsed": 4241,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "3a523cba-853c-4503-be17-9ae6ecd4d5c4"
# !! }}
# Using the HuggingFace library to load the dataset
train_dataset = load_dataset("parquet", data_files={'train': f'{base_path}V2/002_Model/training_data/train_data.parquet'})
val_dataset = load_dataset("parquet", data_files={'val': f'{base_path}V2/002_Model/training_data/val_data.parquet'})

# %%
# !! {"metadata":{
# !!   "id": "2n2bxA7RrahM",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287809990,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "2b14b5cc-dfaa-4ab6-d038-1bea73b62920"
# !! }}
train_dataset


# %%
# !! {"metadata":{
# !!   "id": "nEQOI3Qii_u3",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 466
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287812345,
# !!     "user_tz": 180,
# !!     "elapsed": 2361,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "7e5d6b2e-6199-4ac4-c093-2041c03ed9fb"
# !! }}
pd.DataFrame(train_dataset).head()

# %%
# !! {"metadata":{
# !!   "id": "8plzMGFlri8z",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287812346,
# !!     "user_tz": 180,
# !!     "elapsed": 7,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "059c8a2d-6ce9-44a8-cddf-27b1eb507bec"
# !! }}
val_dataset

# %%
# !! {"metadata":{
# !!   "id": "6aed9ded",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287812346,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
MAX_LEN = 256

def preprocess_function(examples):
    return tokenizer(examples["processed_text"], truncation=True, padding=True,
                     max_length=MAX_LEN)

# %%
# !! {"metadata":{
# !!   "id": "qTvbeMuT4pBO",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 177,
# !!     "referenced_widgets": [
# !!       "1294193998e848449757061cb0d07d66",
# !!       "c238c6bf229c4c61b46edae064d6af51",
# !!       "aa65f67f6f274a67909063087c991af9",
# !!       "5d365283376a4f3c885a724148b39a0c",
# !!       "7c2e7836f97146cdbb300d48e70b04a0",
# !!       "0ed73e6eedda4b8dab64d0fad4597d14",
# !!       "5183c017563d4c44897079e4372168d1",
# !!       "46032c6dbacf450f8411eeaf8b52d904",
# !!       "13fcc5ce5e2346cf99b29515a35e0b7c",
# !!       "fbc2138e02d041f79435a2a803ffa35e",
# !!       "bc76814da6e24651b199e30279e34c00"
# !!     ]
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287822793,
# !!     "user_tz": 180,
# !!     "elapsed": 10451,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "bb8ab066-b156-4e1f-a4cf-de78faaeebb4"
# !! }}
train_dataset.map(preprocess_function, batched=False)

# %%
# !! {"metadata":{
# !!   "id": "47ec0201",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287823197,
# !!     "user_tz": 180,
# !!     "elapsed": 408,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
# Tokenizing the train dataset
tokenized_train_data = train_dataset.map(preprocess_function, batched=False)

# %%
# !! {"metadata":{
# !!   "id": "gCTdVbPArw4H",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287823197,
# !!     "user_tz": 180,
# !!     "elapsed": 13,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "2cc83d53-ca62-4463-84ec-2692e6b53c6a"
# !! }}
tokenized_train_data

# %%
# !! {"metadata":{
# !!   "id": "d4cf74ed",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287823800,
# !!     "user_tz": 180,
# !!     "elapsed": 613,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "d822db80-f785-4168-c7dd-ee4ef3b6a702"
# !! }}
tokenized_train_data.cleanup_cache_files()
tokenized_train_data

# %%
# !! {"metadata":{
# !!   "id": "7e4891f7",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 49,
# !!     "referenced_widgets": [
# !!       "11dd9264cf7142b0b01beaf251f00fb7",
# !!       "1882515b3c9e42f6a943edf1a014b2a2",
# !!       "bc53f8c6da73455590c07e528aff21e2",
# !!       "01bd07c685a44152a86b346c6cfae925",
# !!       "73039030d81f4ff7bc54be51bc1435fe",
# !!       "b9d193cefee54da48675adeda0922144",
# !!       "5c468533fcd745a8b061be4bf534e73f",
# !!       "931608f5fe1046c2a6d471250b119d8b",
# !!       "080b6abb32d14ea29491877a30855c08",
# !!       "acad48f123714bf9bf0f5b43766a6a2a",
# !!       "a449fdc68ce54daaaa29f00fd9aa0a88"
# !!     ]
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287826025,
# !!     "user_tz": 180,
# !!     "elapsed": 2241,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "885d1b21-15a0-4451-ba32-bd2a65a18175"
# !! }}
# Tokenizing the validation dataset
tokenized_val_data = val_dataset.map(preprocess_function, batched=False)

# %%
# !! {"metadata":{
# !!   "id": "jKvDkm_Fr5yM",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287826025,
# !!     "user_tz": 180,
# !!     "elapsed": 8,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "5d5677f3-e247-4b77-a0f4-25ed0baf573e"
# !! }}
tokenized_val_data

# %%
# !! {"metadata":{
# !!   "id": "302110cc",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287826293,
# !!     "user_tz": 180,
# !!     "elapsed": 273,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "4da5324b-6b57-429b-bab5-668d3ae89a98"
# !! }}
tokenized_val_data.cleanup_cache_files()

# %%
# !! {"metadata":{
# !!   "id": "155ced38"
# !! }}
"""
### Creating the model
"""

# %%
# !! {"metadata":{
# !!   "id": "618e5d8b",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287826293,
# !!     "user_tz": 180,
# !!     "elapsed": 3,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
# Hyperparameters to tune

batch_size = 256  # Reduced from 512
#gradient_accumulation_steps = 16  # Accumulate gradients over 4 steps
num_epochs = 12 # reduced from 15
batches_per_epoch = len(tokenized_train_data["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)

# %%
# !! {"metadata":{
# !!   "id": "fe391be0",
# !!   "scrolled": true,
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287827181,
# !!     "user_tz": 180,
# !!     "elapsed": 593,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "7566a643-0eb1-4881-e4ef-95ed4391d214"
# !! }}
# Allow for use of multiple GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors='tf')

    # Turning dataset into TF dataset
    tf_train_dataset = tokenized_train_data["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "label"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator)

    tf_val_dataset = tokenized_val_data["val"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "label"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator)


print(tf_train_dataset)
print(tf_val_dataset)




# %%
# !! {"metadata":{
# !!   "id": "UXErlehhhtqQ",
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 252,
# !!     "referenced_widgets": [
# !!       "9a0db7fc3a6049b9ac90eef889a4f5ce",
# !!       "c9ec22c5d4d343df8c4bcf8f804d2932",
# !!       "3bf7f73cd5e948b59263222504d1627a",
# !!       "9356df24db954ed2a4b8d6380ed9c246",
# !!       "2e24e2a122d74628b486d1b9681145d2",
# !!       "a77ffa5ab46b43939d590cbaa20ac5f8",
# !!       "fd74893a45284109b7e11db691de9ff5",
# !!       "77f4cd28a3fa4a949e679b9a3266b0fa",
# !!       "c7cace358cac401fa0fe5e6d00c5d4d9",
# !!       "399430c4af4c466ab3f03fcb5c36c107",
# !!       "fb7053bb71634af08203fc2b4b877b1e"
# !!     ]
# !!   },
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732287832646,
# !!     "user_tz": 180,
# !!     "elapsed": 5467,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "1412994c-a7da-42cc-e110-a3711ee6cf90"
# !! }}
 # Using HuggingFace library to create optimizer
lr_scheduler = PolynomialDecay(
initial_learning_rate=5e-5, end_learning_rate=5e-7,
decay_steps=total_train_steps)

    # Loading the DistilBERT model and weights with a classification head
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                     num_labels=len(affiliation_vocab))

    # Create the Adam optimizer within the strategy scope

optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler)


    #model.compile(optimizer=optimizer, # Pass 'adam' as a string
                  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Explicitly define the loss function
     ##             metrics=['accuracy']) # Explicitly define the metrics

      # Loading the DistilBERT model and weights with a classification head
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                 num_labels=len(affiliation_vocab))
model.compile(optimizer=optimizer,metrics=['accuracy'])

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "5420b47b",
# !!   "scrolled": true,
# !!   "outputId": "4275f8f2-5154-491d-d8cc-9a53e6933cb0"
# !! }}
model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=num_epochs)

# %%
# !! {"metadata":{
# !!   "id": "7cba4069"
# !! }}
tf_save_directory = f"{base_path}V2/002_Model/Result_model_lang"

# %%
# !! {"metadata":{
# !!   "id": "a6261844"
# !! }}
# Saving the model, tokenizer, and affiliation (target) vocab
tokenizer.save_pretrained(tf_save_directory)

model.save_pretrained(tf_save_directory)
with open(f"{tf_save_directory}/vocab.pkl", "wb") as f:
    pickle.dump(affiliation_vocab, f)

# %%
# !! {"metadata":{
# !!   "id": "2d2089d2"
# !! }}
model

# %%
# !! {"metadata":{
# !!   "id": "Ab9qL6Xeax71"
# !! }}
sub_p_res = subprocess.run(['pip', 'install', 'colab-convert'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['colab-convert', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/002b_language_model.ipynb', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/002b_language_model.p'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
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
# !!   },
# !!   "widgets": {
# !!     "application/vnd.jupyter.widget-state+json": {
# !!       "6c231dc7b1ff449f87b1486da2591276": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_4bc2ffd625314bf18a84c96e9f515e9e",
# !!             "IPY_MODEL_ed57b06e330c47638b6436674ffaa16e",
# !!             "IPY_MODEL_49e1e838a66945c0894816c17e8d86d3"
# !!           ],
# !!           "layout": "IPY_MODEL_c7268ca62435481cb3ab821611be3f1e"
# !!         }
# !!       },
# !!       "4bc2ffd625314bf18a84c96e9f515e9e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_1e26d73c6ad54e0b91868f39816d4a3b",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_653f505efad54d67a5e8f3b2f2b4ac2e",
# !!           "value": "tokenizer_config.json:\u2007100%"
# !!         }
# !!       },
# !!       "ed57b06e330c47638b6436674ffaa16e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_4060fa244c70445f8a4f88a8a6f21919",
# !!           "max": 48,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_6874fb7198324e2d9b421c50b6675035",
# !!           "value": 48
# !!         }
# !!       },
# !!       "49e1e838a66945c0894816c17e8d86d3": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_90d24853607143c1982652abfe2fcb9b",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_7c53e5a289fa47d983f2d1b4c08a5501",
# !!           "value": "\u200748.0/48.0\u2007[00:00&lt;00:00,\u20072.69kB/s]"
# !!         }
# !!       },
# !!       "c7268ca62435481cb3ab821611be3f1e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "1e26d73c6ad54e0b91868f39816d4a3b": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "653f505efad54d67a5e8f3b2f2b4ac2e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "4060fa244c70445f8a4f88a8a6f21919": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "6874fb7198324e2d9b421c50b6675035": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "90d24853607143c1982652abfe2fcb9b": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "7c53e5a289fa47d983f2d1b4c08a5501": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "c89531dd3eec4d978dc9eb7fd34d08e3": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_2943023c02d44014bdc2c0ae7969fc04",
# !!             "IPY_MODEL_5cb4f065030e47708dcf98a89fae95bc",
# !!             "IPY_MODEL_c8f8b0950a98473da5c62e67d372acf8"
# !!           ],
# !!           "layout": "IPY_MODEL_ee95e5b85e394395abc307a5e060c21b"
# !!         }
# !!       },
# !!       "2943023c02d44014bdc2c0ae7969fc04": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_a8d998e0105f47a6aa0a653365ad5139",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_87272c0ea53145b99e59f465da032708",
# !!           "value": "vocab.txt:\u2007100%"
# !!         }
# !!       },
# !!       "5cb4f065030e47708dcf98a89fae95bc": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_cca92b3179604d20a9fbcd0bb0e3755e",
# !!           "max": 231508,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_b220022e8f204e27a12105f5dbb52d60",
# !!           "value": 231508
# !!         }
# !!       },
# !!       "c8f8b0950a98473da5c62e67d372acf8": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_468b61c5bb084a9294c600d911b3d7be",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_f300d22a64ef48aea1b22a5d071b522c",
# !!           "value": "\u2007232k/232k\u2007[00:00&lt;00:00,\u20079.67MB/s]"
# !!         }
# !!       },
# !!       "ee95e5b85e394395abc307a5e060c21b": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "a8d998e0105f47a6aa0a653365ad5139": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "87272c0ea53145b99e59f465da032708": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "cca92b3179604d20a9fbcd0bb0e3755e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "b220022e8f204e27a12105f5dbb52d60": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "468b61c5bb084a9294c600d911b3d7be": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "f300d22a64ef48aea1b22a5d071b522c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "93dd09198cb443e3ad1884632ebac01b": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_5fd6701b6e2b49c89879f2b88125343a",
# !!             "IPY_MODEL_205b984723c34b3da8e69f5318c443de",
# !!             "IPY_MODEL_28847f4bb0c44fa0837b5ebd58df5c01"
# !!           ],
# !!           "layout": "IPY_MODEL_0c5e697fb80e4619808beaf0fe56f6cb"
# !!         }
# !!       },
# !!       "5fd6701b6e2b49c89879f2b88125343a": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_f0b5354c481c4494a572d027cea8e874",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_70ade11ba70a40c08418cb8c89154c19",
# !!           "value": "tokenizer.json:\u2007100%"
# !!         }
# !!       },
# !!       "205b984723c34b3da8e69f5318c443de": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_a0d253bf7d3245639bb3523f64c8c8b1",
# !!           "max": 466062,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_8e3af37de70b4f1fb44225065c25076f",
# !!           "value": 466062
# !!         }
# !!       },
# !!       "28847f4bb0c44fa0837b5ebd58df5c01": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_f529487d6f0b454b854f9fd550796f2b",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_5b8212d3d80f432b8cbfe9bb87f2e45b",
# !!           "value": "\u2007466k/466k\u2007[00:00&lt;00:00,\u200719.5MB/s]"
# !!         }
# !!       },
# !!       "0c5e697fb80e4619808beaf0fe56f6cb": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "f0b5354c481c4494a572d027cea8e874": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "70ade11ba70a40c08418cb8c89154c19": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "a0d253bf7d3245639bb3523f64c8c8b1": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "8e3af37de70b4f1fb44225065c25076f": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "f529487d6f0b454b854f9fd550796f2b": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "5b8212d3d80f432b8cbfe9bb87f2e45b": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "4d91076cb18245b09f146b0099da5563": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_c5c909db859c41a59c883c077f9c3998",
# !!             "IPY_MODEL_b2b4d86760614ca2aeca5439c8f3f50d",
# !!             "IPY_MODEL_225b65f82ccd4a92a0d36b24ce9d5130"
# !!           ],
# !!           "layout": "IPY_MODEL_871b7cb82d1140c1b0e7c96c93ba40bf"
# !!         }
# !!       },
# !!       "c5c909db859c41a59c883c077f9c3998": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_7b8936db9f284c648802fff2f0448432",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_b32f0110a9214e7cb3c34af64356f754",
# !!           "value": "config.json:\u2007100%"
# !!         }
# !!       },
# !!       "b2b4d86760614ca2aeca5439c8f3f50d": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_da2ab0cfb7b14787bcb659f456213cb5",
# !!           "max": 483,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_12938abc05a74e359064114a0896607c",
# !!           "value": 483
# !!         }
# !!       },
# !!       "225b65f82ccd4a92a0d36b24ce9d5130": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_7de0dfd525364ed19c752677ecdddc89",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_2c7428c22d2e464c928d77e443b9d55a",
# !!           "value": "\u2007483/483\u2007[00:00&lt;00:00,\u200723.4kB/s]"
# !!         }
# !!       },
# !!       "871b7cb82d1140c1b0e7c96c93ba40bf": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "7b8936db9f284c648802fff2f0448432": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "b32f0110a9214e7cb3c34af64356f754": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "da2ab0cfb7b14787bcb659f456213cb5": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "12938abc05a74e359064114a0896607c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "7de0dfd525364ed19c752677ecdddc89": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "2c7428c22d2e464c928d77e443b9d55a": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "e8dda68557e9440d8209cc0bf3ff8c64": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_9202a20b880648259844aafce6f7e020",
# !!             "IPY_MODEL_7aee38a33a1f4ae297f63128e4bc9650",
# !!             "IPY_MODEL_5dbe7f1255484136be03d3a4f051b691"
# !!           ],
# !!           "layout": "IPY_MODEL_15f1fb12ff56436cae5b1e239556d11f"
# !!         }
# !!       },
# !!       "9202a20b880648259844aafce6f7e020": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_f480f34659de4023b8f97b9433e65695",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_3172052759934d6ab248500bbb373946",
# !!           "value": "Generating\u2007train\u2007split:\u2007"
# !!         }
# !!       },
# !!       "7aee38a33a1f4ae297f63128e4bc9650": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_2da6f945f6dc46e68ab86ffbb7ba8902",
# !!           "max": 1,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_a05cc81428744913a2886051b64ba3d6",
# !!           "value": 1
# !!         }
# !!       },
# !!       "5dbe7f1255484136be03d3a4f051b691": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_c88a4137f3e043f5a7c98da1f99968ba",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_d2d393baff4b40a4b49294fd3e7a3af4",
# !!           "value": "\u200716487/0\u2007[00:00&lt;00:00,\u200722496.67\u2007examples/s]"
# !!         }
# !!       },
# !!       "15f1fb12ff56436cae5b1e239556d11f": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "f480f34659de4023b8f97b9433e65695": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "3172052759934d6ab248500bbb373946": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "2da6f945f6dc46e68ab86ffbb7ba8902": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": "20px"
# !!         }
# !!       },
# !!       "a05cc81428744913a2886051b64ba3d6": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "c88a4137f3e043f5a7c98da1f99968ba": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "d2d393baff4b40a4b49294fd3e7a3af4": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "0c67b99e3b5b4b87b54d91f0c72c3685": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_f95227c3f3674b4c9343b61a38a132f3",
# !!             "IPY_MODEL_936fae792f2b45b38430b24182728429",
# !!             "IPY_MODEL_3fae53959f604744bde9cd11957b81ce"
# !!           ],
# !!           "layout": "IPY_MODEL_4faa798380b5419584cebc97b1dcc57d"
# !!         }
# !!       },
# !!       "f95227c3f3674b4c9343b61a38a132f3": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_e5e8a9f17dfa4d1eaf00560f57e369fa",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_072c35420af94b5ba08ee675cd20122f",
# !!           "value": "Generating\u2007val\u2007split:\u2007"
# !!         }
# !!       },
# !!       "936fae792f2b45b38430b24182728429": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_d93f566d4bf6487fb4ef56eaf4b489d1",
# !!           "max": 1,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_246f621978d34c60adae1d2c1dbb84a2",
# !!           "value": 1
# !!         }
# !!       },
# !!       "3fae53959f604744bde9cd11957b81ce": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_1e44b863cfd14b14af98548199b2ceb2",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_5428080a263f4947aa0419558a106f97",
# !!           "value": "\u20074122/0\u2007[00:00&lt;00:00,\u200765322.80\u2007examples/s]"
# !!         }
# !!       },
# !!       "4faa798380b5419584cebc97b1dcc57d": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "e5e8a9f17dfa4d1eaf00560f57e369fa": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "072c35420af94b5ba08ee675cd20122f": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "d93f566d4bf6487fb4ef56eaf4b489d1": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": "20px"
# !!         }
# !!       },
# !!       "246f621978d34c60adae1d2c1dbb84a2": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "1e44b863cfd14b14af98548199b2ceb2": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "5428080a263f4947aa0419558a106f97": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "1294193998e848449757061cb0d07d66": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_c238c6bf229c4c61b46edae064d6af51",
# !!             "IPY_MODEL_aa65f67f6f274a67909063087c991af9",
# !!             "IPY_MODEL_5d365283376a4f3c885a724148b39a0c"
# !!           ],
# !!           "layout": "IPY_MODEL_7c2e7836f97146cdbb300d48e70b04a0"
# !!         }
# !!       },
# !!       "c238c6bf229c4c61b46edae064d6af51": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_0ed73e6eedda4b8dab64d0fad4597d14",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_5183c017563d4c44897079e4372168d1",
# !!           "value": "Map:\u2007100%"
# !!         }
# !!       },
# !!       "aa65f67f6f274a67909063087c991af9": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_46032c6dbacf450f8411eeaf8b52d904",
# !!           "max": 16487,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_13fcc5ce5e2346cf99b29515a35e0b7c",
# !!           "value": 16487
# !!         }
# !!       },
# !!       "5d365283376a4f3c885a724148b39a0c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_fbc2138e02d041f79435a2a803ffa35e",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_bc76814da6e24651b199e30279e34c00",
# !!           "value": "\u200716487/16487\u2007[00:10&lt;00:00,\u20071990.66\u2007examples/s]"
# !!         }
# !!       },
# !!       "7c2e7836f97146cdbb300d48e70b04a0": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "0ed73e6eedda4b8dab64d0fad4597d14": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "5183c017563d4c44897079e4372168d1": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "46032c6dbacf450f8411eeaf8b52d904": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "13fcc5ce5e2346cf99b29515a35e0b7c": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "fbc2138e02d041f79435a2a803ffa35e": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "bc76814da6e24651b199e30279e34c00": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "11dd9264cf7142b0b01beaf251f00fb7": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_1882515b3c9e42f6a943edf1a014b2a2",
# !!             "IPY_MODEL_bc53f8c6da73455590c07e528aff21e2",
# !!             "IPY_MODEL_01bd07c685a44152a86b346c6cfae925"
# !!           ],
# !!           "layout": "IPY_MODEL_73039030d81f4ff7bc54be51bc1435fe"
# !!         }
# !!       },
# !!       "1882515b3c9e42f6a943edf1a014b2a2": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_b9d193cefee54da48675adeda0922144",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_5c468533fcd745a8b061be4bf534e73f",
# !!           "value": "Map:\u2007100%"
# !!         }
# !!       },
# !!       "bc53f8c6da73455590c07e528aff21e2": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_931608f5fe1046c2a6d471250b119d8b",
# !!           "max": 4122,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_080b6abb32d14ea29491877a30855c08",
# !!           "value": 4122
# !!         }
# !!       },
# !!       "01bd07c685a44152a86b346c6cfae925": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_acad48f123714bf9bf0f5b43766a6a2a",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_a449fdc68ce54daaaa29f00fd9aa0a88",
# !!           "value": "\u20074122/4122\u2007[00:02&lt;00:00,\u20071976.81\u2007examples/s]"
# !!         }
# !!       },
# !!       "73039030d81f4ff7bc54be51bc1435fe": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "b9d193cefee54da48675adeda0922144": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "5c468533fcd745a8b061be4bf534e73f": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "931608f5fe1046c2a6d471250b119d8b": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "080b6abb32d14ea29491877a30855c08": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "acad48f123714bf9bf0f5b43766a6a2a": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "a449fdc68ce54daaaa29f00fd9aa0a88": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "9a0db7fc3a6049b9ac90eef889a4f5ce": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HBoxModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HBoxModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HBoxView",
# !!           "box_style": "",
# !!           "children": [
# !!             "IPY_MODEL_c9ec22c5d4d343df8c4bcf8f804d2932",
# !!             "IPY_MODEL_3bf7f73cd5e948b59263222504d1627a",
# !!             "IPY_MODEL_9356df24db954ed2a4b8d6380ed9c246"
# !!           ],
# !!           "layout": "IPY_MODEL_2e24e2a122d74628b486d1b9681145d2"
# !!         }
# !!       },
# !!       "c9ec22c5d4d343df8c4bcf8f804d2932": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_a77ffa5ab46b43939d590cbaa20ac5f8",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_fd74893a45284109b7e11db691de9ff5",
# !!           "value": "model.safetensors:\u2007100%"
# !!         }
# !!       },
# !!       "3bf7f73cd5e948b59263222504d1627a": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "FloatProgressModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "FloatProgressModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "ProgressView",
# !!           "bar_style": "success",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_77f4cd28a3fa4a949e679b9a3266b0fa",
# !!           "max": 267954768,
# !!           "min": 0,
# !!           "orientation": "horizontal",
# !!           "style": "IPY_MODEL_c7cace358cac401fa0fe5e6d00c5d4d9",
# !!           "value": 267954768
# !!         }
# !!       },
# !!       "9356df24db954ed2a4b8d6380ed9c246": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "HTMLModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_dom_classes": [],
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "HTMLModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/controls",
# !!           "_view_module_version": "1.5.0",
# !!           "_view_name": "HTMLView",
# !!           "description": "",
# !!           "description_tooltip": null,
# !!           "layout": "IPY_MODEL_399430c4af4c466ab3f03fcb5c36c107",
# !!           "placeholder": "\u200b",
# !!           "style": "IPY_MODEL_fb7053bb71634af08203fc2b4b877b1e",
# !!           "value": "\u2007268M/268M\u2007[00:01&lt;00:00,\u2007247MB/s]"
# !!         }
# !!       },
# !!       "2e24e2a122d74628b486d1b9681145d2": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "a77ffa5ab46b43939d590cbaa20ac5f8": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "fd74893a45284109b7e11db691de9ff5": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "77f4cd28a3fa4a949e679b9a3266b0fa": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "c7cace358cac401fa0fe5e6d00c5d4d9": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "ProgressStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "ProgressStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "bar_color": null,
# !!           "description_width": ""
# !!         }
# !!       },
# !!       "399430c4af4c466ab3f03fcb5c36c107": {
# !!         "model_module": "@jupyter-widgets/base",
# !!         "model_name": "LayoutModel",
# !!         "model_module_version": "1.2.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/base",
# !!           "_model_module_version": "1.2.0",
# !!           "_model_name": "LayoutModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "LayoutView",
# !!           "align_content": null,
# !!           "align_items": null,
# !!           "align_self": null,
# !!           "border": null,
# !!           "bottom": null,
# !!           "display": null,
# !!           "flex": null,
# !!           "flex_flow": null,
# !!           "grid_area": null,
# !!           "grid_auto_columns": null,
# !!           "grid_auto_flow": null,
# !!           "grid_auto_rows": null,
# !!           "grid_column": null,
# !!           "grid_gap": null,
# !!           "grid_row": null,
# !!           "grid_template_areas": null,
# !!           "grid_template_columns": null,
# !!           "grid_template_rows": null,
# !!           "height": null,
# !!           "justify_content": null,
# !!           "justify_items": null,
# !!           "left": null,
# !!           "margin": null,
# !!           "max_height": null,
# !!           "max_width": null,
# !!           "min_height": null,
# !!           "min_width": null,
# !!           "object_fit": null,
# !!           "object_position": null,
# !!           "order": null,
# !!           "overflow": null,
# !!           "overflow_x": null,
# !!           "overflow_y": null,
# !!           "padding": null,
# !!           "right": null,
# !!           "top": null,
# !!           "visibility": null,
# !!           "width": null
# !!         }
# !!       },
# !!       "fb7053bb71634af08203fc2b4b877b1e": {
# !!         "model_module": "@jupyter-widgets/controls",
# !!         "model_name": "DescriptionStyleModel",
# !!         "model_module_version": "1.5.0",
# !!         "state": {
# !!           "_model_module": "@jupyter-widgets/controls",
# !!           "_model_module_version": "1.5.0",
# !!           "_model_name": "DescriptionStyleModel",
# !!           "_view_count": null,
# !!           "_view_module": "@jupyter-widgets/base",
# !!           "_view_module_version": "1.2.0",
# !!           "_view_name": "StyleView",
# !!           "description_width": ""
# !!         }
# !!       }
# !!     }
# !!   }
# !! }}
