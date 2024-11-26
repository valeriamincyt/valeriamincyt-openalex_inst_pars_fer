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
# !!   "id": "49aa7c54",
# !!   "outputId": "912b8148-00d8-4653-89b9-319d907c4e9f",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289819027,
# !!     "user_tz": 180,
# !!     "elapsed": 10180,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
import os
import re
import json
sub_p_res = subprocess.run(['pip', 'install', 'boto3'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'pickle'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'collections'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'unidecode'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['pip', 'install', 'langdetect'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
import boto3
import pickle
from unidecode import unidecode
from collections import Counter
from langdetect import detect
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 500)
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "Jz1fT1jBmKk3",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289822732,
# !!     "user_tz": 180,
# !!     "elapsed": 3710,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "86e4b7d1-4766-4f24-a0ec-40a686b7a3a1"
# !! }}
from google.colab import drive
from google.colab import auth
drive.mount('/content/drive', force_remount=True)

# %%
# !! {"metadata":{
# !!   "id": "yJQRyf6baXhl"
# !! }}
"""
**Cambiar rutas**
"""

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "RHyewZl6lw2Z",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289822733,
# !!     "user_tz": 180,
# !!     "elapsed": 56,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "96b8d18b-c9ff-4fb0-80da-c3e3a00c9890"
# !! }}
# Define the path
#model_path = "./path/to/model/artifacts/"
base_path="/content/drive/MyDrive/openalex-institution-parsing/"
artifacts_path = f"{base_path}Crudos/institution_tagger_v2_artifacts/"
model_path = f"{base_path}V2/002_Model/"


# Load the needed files
with open(os.path.join(artifacts_path, "departments_list1.pkl"), "rb") as f:
    departments_list = pickle.load(f)

print("Loaded list of departments")
print(departments_list)

with open(os.path.join(artifacts_path, "full_affiliation_dict1.pkl"), "rb") as f:
    full_affiliation_dict = pickle.load(f)

print("Loaded affiliation dictionary")
print(full_affiliation_dict)
print(len(full_affiliation_dict))


with open(os.path.join(artifacts_path, "multi_inst_names_ids1.pkl"), "rb") as f:
    multi_inst_names_ids = pickle.load(f)

print("Loaded list of institutions that have common name with other institutions.")

with open(os.path.join(artifacts_path, "countries_list_flat1.pkl"), "rb") as f:
    countries_list_flat = pickle.load(f)
print("Loaded flat list of countries")
print(countries_list_flat)

with open(os.path.join(artifacts_path, "countries.json"), "r") as f:
    countries_dict = json.load(f)

print("Loaded countries dictionary")
print(countries_dict
      )
with open(os.path.join(artifacts_path, "city_country_list1.pkl"), "rb") as f:
    city_country_list = pickle.load(f)

print("Loaded strings of city/country combinations")
print(city_country_list)

with open(os.path.join(artifacts_path, "affiliation_vocab_argentina.pkl"), "rb") as f:
    affiliation_vocab = pickle.load(f)

affiliation_vocab = {int(i):int(j) for i,j in affiliation_vocab.items()}

#se agrega la diccionario el valor -1

#affiliation_vocab[-1]=len(affiliation_vocab)

inverse_affiliation_vocab = {(i):(j) for j,i in affiliation_vocab.items()}

print("Loaded affiliation vocab")

##inverse_affiliation_vocab[-1]=len(inverse_affiliation_vocab)

print("affiliation_vocab_argentina")
print(affiliation_vocab)

print("inverse_affiliation_vocab")
print(inverse_affiliation_vocab)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "8lFx-URCTN2E",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289822733,
# !!     "user_tz": 180,
# !!     "elapsed": 48,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "7274299f-4318-4262-96ca-8c4aa1635f78"
# !! }}
len(full_affiliation_dict)

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "qjL94XVJYX3C",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289823090,
# !!     "user_tz": 180,
# !!     "elapsed": 363,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "eb7d23f7-2c03-4822-a46e-a8481bb1717f"
# !! }}
full_affiliation_dict

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "TYtuQEoQ_XHL",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826701,
# !!     "user_tz": 180,
# !!     "elapsed": 3614,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "a1c549f3-10b8-4ebb-a849-7c9b2d2ed752"
# !! }}
# Load the tokenizers
language_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", return_tensors='tf')
data_collator = DataCollatorWithPadding(tokenizer=language_tokenizer,
                                        return_tensors='tf')

#basic_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, "basic_model_tokenizer"))
basic_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, "basic_model_tokenizer"))

# Load the models
#language_model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "language_model"))
language_model = TFAutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "Result_model_lang"))
language_model.trainable = False

from tensorflow.keras.layers import TFSMLayer
basic_path=f'{base_path}V2/002_Model/Result_basic_model/'
##TFSMLayer Reload a Keras model/layer that was saved via SavedModel / ExportArchive.
basic_model = TFSMLayer(os.path.join(basic_path, "basic_model"), call_endpoint='serving_default')

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "33bvIVsdqW2R",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826701,
# !!     "user_tz": 180,
# !!     "elapsed": 21,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "57a9c8d7-e78d-4d83-caa3-9c5807665521"
# !! }}
countries_list_flat[0:3]

# %%
# !! {"metadata":{
# !!   "id": "e1695a97",
# !!   "scrolled": true,
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826702,
# !!     "user_tz": 180,
# !!     "elapsed": 20,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
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
    text = text.replace("U.S. Army", "United States Army")
    text = text.replace("U.S. Navy", "United States Navy")
    text = text.replace("U.S. Air Force", "United States Air Force")

    # take out spaces, commas, dashes, periods, etcs
    text = re.sub("[^0-9a-zA-Z]", "", text)

    return text

def get_country_in_string(text):
    """
    Looks for countries in the affiliation string to be used in filtering later on.
    """
    countries_in_string = []
    _ = [countries_in_string.append(x) for x,y in countries_dict.items() if
         np.max([1 if re.search(fr"\b{i}\b", text) else 0 for i in y]) > 0]
    _ = [countries_in_string.append(x) for x,y in countries_dict.items() if
         np.max([1 if re.search(fr"\b{i}\b", text.replace(".","")) else 0 for i in y]) > 0]
    return list(set(countries_in_string))

def max_len_and_pad(tok_sent):
    """
    Processes the basic model data to the correct input length.
    """
    max_len = 128
    tok_sent = tok_sent[:max_len]
    tok_sent = tok_sent + [0]*(max_len - len(tok_sent))
    return tok_sent


def get_language(orig_aff_string):
    """
    Guesses the language of the affiliation string to be used for filtering later.
    """
    try:
        string_lang = detect(orig_aff_string)
    except:
        string_lang = 'en'

    return string_lang

def get_initial_pred(orig_aff_string, string_lang, countries_in_string, comma_split_len):
    """
    Initial hard-coded filtering of the affiliation text to ensure that meaningless strings
    and strings in other languages are not given an institution.
    """
    if string_lang in ['fa','ko','zh-cn','zh-tw','ja','uk','ru','vi','ar']:
        init_pred = None
    elif len(string_match_clean(str(orig_aff_string))) <=2:
        init_pred = None
    elif ((orig_aff_string.startswith("Dep") |
           orig_aff_string.startswith("School") |
           orig_aff_string.startswith("Ministry")) &
          (comma_split_len < 2) &
          (not countries_in_string)):
        init_pred = None
    elif orig_aff_string in departments_list:
        init_pred = None
    elif string_match_clean(str(orig_aff_string).strip()) in city_country_list:
        init_pred = None
    elif re.search(r"\b(LIANG|YANG|LIU|XIE|JIA|ZHANG)\b",
                   orig_aff_string):
        for inst_name in ["Hospital","University","School","Academy","Institute",
                          "Ministry","Laboratory","College"]:
            if inst_name in str(orig_aff_string):
                init_pred = 0
                break
            else:
                init_pred = None

    elif re.search(r"\b(et al)\b", orig_aff_string):
        if str(orig_aff_string).strip().endswith('et al'):
            init_pred = None
        else:
            init_pred = 0
    else:
        init_pred = 0
    return init_pred

def get_language_model_prediction(decoded_text, all_countries):
    """
    Preprocesses the decoded text and gets the output labels and scores for the language model.
    """
    lang_tok_data = language_tokenizer(decoded_text, truncation=True, padding=True, max_length=512)

    data = data_collator(lang_tok_data)
    all_scores, all_labels = tf.math.top_k(tf.nn.softmax(
            language_model.predict([data['input_ids'],
                                    data['attention_mask']]).logits).numpy(), 20)

    all_scores = all_scores.numpy().tolist()
    all_labels = all_labels.numpy().tolist()

    final_preds_scores = []
    for scores, labels, countries in zip(all_scores, all_labels, all_countries):
        final_pred, final_score, mapping = get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                         affiliation_vocab,
                                                                         inverse_affiliation_vocab)
        final_preds_scores.append([final_pred, final_score, mapping])

    return final_preds_scores

def get_final_basic_or_language_model_pred(scores, labels, countries, vocab, inv_vocab):
    """
    Takes the scores and labels from either model and performs a quick country matching
    to see if the country found in the string can be matched to the country of the
    predicted institution.
    """
    mapped_labels = [inv_vocab[i] for i,j in zip(labels,scores) if i!=vocab[-1]]
    scores = [j for i,j in zip(labels,scores) if i!=vocab[-1]]
    final_pred = mapped_labels[0]
    final_score = scores[0]
    if not full_affiliation_dict[mapped_labels[0]]['country']:
        pass
    else:
        if not countries:
            pass
        else:
            for pred,score in zip(mapped_labels, scores):
                if not full_affiliation_dict[pred]['country']:
                    # trying pass instead of break to give time to find the correct country
                    pass
                elif full_affiliation_dict[pred]['country'] in countries:
                    final_pred = pred
                    final_score = score
                    break
                else:
                    pass
    return final_pred, final_score, mapped_labels

def get_similar_preds_to_remove(decoded_string, curr_preds):
    """
    Looks for organizations with similar/matching names and only predicts for one of those organizations.
    """
    preds_to_remove = []
    pred_display_names = [full_affiliation_dict[i]['display_name'] for i in curr_preds]
    counts_of_preds = Counter(pred_display_names)

    preds_array = np.array(curr_preds)
    preds_names_array = np.array(pred_display_names)

    for pred_name in counts_of_preds.items():
        temp_preds_to_remove = []
        to_use = []
        if pred_name[1] > 1:
            list_to_check = preds_array[preds_names_array == pred_name[0]].tolist()
            for pred in list_to_check:
                if string_match_clean(full_affiliation_dict[pred]['city']) in decoded_string:
                    to_use.append(pred)
                else:
                    temp_preds_to_remove.append(pred)
            if not to_use:
                to_use = temp_preds_to_remove[0]
                preds_to_remove += temp_preds_to_remove[1:]
            else:
                preds_to_remove += temp_preds_to_remove
        else:
            pass

    return preds_to_remove


def check_for_city_and_country_in_string(raw_sentence, countries, aff_dict_entry):
    """
    Checks for city and country and string for a common name institution.
    """
    if (aff_dict_entry['country'] in countries) & (aff_dict_entry['city'] in raw_sentence):
        return True
    else:
        return False

# %%
# !! {"metadata":{
# !!   "id": "Nh0IvWABj9Ze",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826702,
# !!     "user_tz": 180,
# !!     "elapsed": 18,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def get_basic_model_prediction(decoded_text, all_countries):
    """
    Preprocesses the decoded text and gets the output labels and scores for the basic model.
    """
    basic_tok_data = basic_tokenizer(decoded_text)['input_ids']
    basic_tok_data = [max_len_and_pad(x) for x in basic_tok_data]

    # Convert the padded data to a TensorFlow tensor
    #basic_tok_data=decoded_text['basic_tok_data'].tolist()
    basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)## cambio

    # Proceed with the rest of your code
    # Get the prediction output from the model
    predictions = basic_model(basic_tok_tensor)

    # Extract the 'output_0' tensor from the predictions dictionary
    model_output = predictions['output_0']

    # Apply tf.math.top_k to the extracted tensor
    all_scores, all_labels = tf.math.top_k(model_output, 3)  # Use the extracted tensor here
    all_scores = all_scores.numpy().tolist()
    all_labels = all_labels.numpy().tolist()

    final_preds_scores = []
    for scores, labels, countries in zip(all_scores, all_labels, all_countries):
        final_pred, final_score, mapping = get_final_basic_or_language_model_pred(scores, labels, countries,
                                                                         affiliation_vocab,
                                                                         inverse_affiliation_vocab)
        final_preds_scores.append([final_pred, final_score, mapping])

    return final_preds_scores

# %%
# !! {"metadata":{
# !!   "id": "zuYwQdAlYgcz"
# !! }}
"""

"""

# %%
# !! {"metadata":{
# !!   "id": "tpku_w8GnMzp",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826702,
# !!     "user_tz": 180,
# !!     "elapsed": 17,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def preprocess_affiliation_text(text):
        # Normalización de caracteres
        # Call the unidecode function within the unidecode module
        text = unidecode(text)
        # Convertir a minúsculas
        text = text.lower()
        # Remover caracteres especiales (si es necesario)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return  text

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "3L9ohscAj07p",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826703,
# !!     "user_tz": 180,
# !!     "elapsed": 15,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "66ce30dd-8164-4fac-a01e-d2a2593eae41"
# !! }}
def raw_data_to_predictions(df, lang_thresh, basic_thresh):
    """
    High level function to go from a raw input dataframe to the final dataframe with affiliation
    ID prediction.
    """
    # Implementing the functions above
    df['lang'] = df['affiliation_string'].apply(get_language)
    df['country_in_string'] = df['affiliation_string'].apply(get_country_in_string)
    df['comma_split_len'] = df['affiliation_string'].apply(lambda x: len([i if i else "" for i in
                                                                          x.split(",")]))

    # Gets initial indicator of whether or not the string should go through the models
    df['affiliation_id'] = df.apply(lambda x: get_initial_pred(x.affiliation_string, x.lang,
                                                               x.country_in_string, x.comma_split_len), axis=1)

    # Filter out strings that won't go through the models
    to_predict = df[df['affiliation_id']==0.0].drop_duplicates(subset=['affiliation_string']).copy()
    to_predict['affiliation_id'] = to_predict['affiliation_id'].astype('int')

    # Decode text so only ASCII characters are used
    to_predict['decoded_text'] = to_predict['affiliation_string'].apply(unidecode)

####agregado
      # Aplicar el preprocesamiento a los datos de entrenamiento, validación y prueba
    to_predict['decoded_text'] = to_predict['affiliation_string'].apply(preprocess_affiliation_text)
    print(to_predict.head(2))
    to_predict=pd.DataFrame(to_predict)
###final agregado


    # Get predictions and scores for each model
    to_predict['lang_pred_score'] = get_language_model_prediction(to_predict['decoded_text'].to_list(),
                                                                  to_predict['country_in_string'].to_list())
    to_predict['basic_pred_score'] = get_basic_model_prediction(to_predict['decoded_text'].to_list(),
                                                                to_predict['country_in_string'].to_list())

    # Get the final prediction for each affiliation string
    to_predict['affiliation_id'] = to_predict.apply(lambda x:
                                                    get_final_prediction(x.basic_pred_score,
                                                                         x.lang_pred_score,
                                                                         x.country_in_string,
                                                                         x.affiliation_string,
                                                                         lang_thresh, basic_thresh), axis=1)

    # Merge predictions to original dataframe to get the same order as the data that was requested
    final_df = df[['affiliation_string']].merge(to_predict[['affiliation_string','affiliation_id']],
                                                how='left', on='affiliation_string')

#     final_df['affiliation_id'] = final_df['affiliation_id'].fillna(-1).astype('int')
    return final_df

print("Models initialized")

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/"
# !!   },
# !!   "id": "ZOlLtFQULtK5",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826994,
# !!     "user_tz": 180,
# !!     "elapsed": 302,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "c2e8369a-e02e-4a98-ec13-bb604d58e686"
# !! }}
def get_final_prediction(basic_pred_score, lang_pred_score, countries, raw_sentence, lang_thresh, basic_thresh):
    """
    Performs the model comparison and filtering to get the final prediction.
    """

    # Getting the individual preds and scores for both models
    pred_lang, score_lang, mapped_lang = lang_pred_score
    pred_basic, score_basic, mapped_basic = basic_pred_score

#     print(f"lang: {pred_lang} - {score_lang}")
#     print(f"basic: {pred_basic} - {score_basic}")

    # Logic for combining the two models

    final_preds = []
    final_scores = []
    final_cats = []
    check_pred = []
    if pred_lang == pred_basic:
        final_preds.append(pred_lang)
        final_scores.append(score_lang)
        final_cats.append('model_match')
        check_pred.append(pred_lang)
    elif score_basic > basic_thresh:
        final_preds.append(pred_basic)
        final_scores.append(score_basic)
        final_cats.append('basic_thresh')
        check_pred.append(pred_basic)
    elif score_lang > lang_thresh:
        final_preds.append(pred_lang)
        final_scores.append(score_lang)
        final_cats.append('lang_thresh')
        check_pred.append(pred_lang)
    elif (score_basic > 0.01) & ('China' in countries) & ('Natural Resource' in raw_sentence):
        final_preds.append(pred_basic)
        final_scores.append(score_basic)
        final_cats.append('basic_thresh_second')
        check_pred.append(pred_basic)
    else:
        final_preds.append(-1)
        final_scores.append(0.0)
        final_cats.append('nothing')

    print(final_preds)
    all_mapped = list(set(mapped_lang + mapped_basic))

    decoded_affiliation_string = string_match_clean(raw_sentence)
    all_mapped_strings = [full_affiliation_dict[i]['final_names'] for i in all_mapped]


    matched_preds = []
    matched_strings = []
#     print(f"RAW: {raw_sentence}")
#     print(f"CLEAN: {decoded_affiliation_string}")
#     print(f"COUNTRIES: {countries}")
    for inst_id, match_strings in zip(all_mapped, all_mapped_strings):
#         print(f"------{full_affiliation_dict[inst_id]['display_name']} - {inst_id}")
        if inst_id not in final_preds:
            for match_string in match_strings:
#                 print(f"------{match_string} ({full_affiliation_dict[inst_id]['display_name']} - {inst_id})")
                if match_string in decoded_affiliation_string:
#                     print("FOUND A MATCH")
                    if not full_affiliation_dict[inst_id]['country']:
#                         print("######match (no country_dict for aff ID)")
                        matched_preds.append(inst_id)
                        matched_strings.append(match_string)
                    elif not countries:
#                         print("######match (no country in string)")
                        if inst_id not in multi_inst_names_ids:
                            matched_preds.append(inst_id)
                            matched_strings.append(match_string)
                        else:
                            pass
                    elif full_affiliation_dict[inst_id]['country'] in countries:
#                         print("######match (country matches string)")
                        matched_preds.append(inst_id)
                        matched_strings.append(match_string)
                    else:
                        pass
                    break
                else:
                    pass
        else:
            pass

    # need to check for institutions that are a subset of another institution
    skip_matching = []
    for inst_id, matched_string in zip(matched_preds, matched_strings):
        for inst_id2, matched_string2 in zip(matched_preds, matched_strings):
            if (matched_string in matched_string2) & (matched_string != matched_string2):
                skip_matching.append(inst_id)

    if check_pred:
        for inst_id, matched_string in zip(matched_preds, matched_strings):
            for final_string in full_affiliation_dict[check_pred[0]]['final_names']:
                if matched_string in final_string:
                    skip_matching.append(inst_id)

    for matched_pred in matched_preds:
        if matched_pred not in skip_matching:
            final_preds.append(matched_pred)
            final_scores.append(0.95)
            final_cats.append('string_match')

    if (final_cats[0] == 'nothing') & (len(final_preds)>1):
        final_preds = final_preds[1:]
        final_scores = final_scores[1:]
        final_cats = final_cats[1:]

    # check if many names belong to same organization name (different locations)
    if (final_preds[0] != -1) & (len(final_preds)>1):
        final_display_names = [full_affiliation_dict[x]['display_name'] for x in final_preds]

        if len(final_display_names) == set(final_display_names):
            pass
        else:
            final_preds_after_removal = []
            final_scores_after_removal = []
            final_cats_after_removal = []
            preds_to_remove = get_similar_preds_to_remove(decoded_affiliation_string, final_preds)
            for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                if temp_pred in preds_to_remove:
                    pass
                else:
                    final_preds_after_removal.append(temp_pred)
                    final_scores_after_removal.append(temp_score)
                    final_cats_after_removal.append(temp_cat)

            final_preds = final_preds_after_removal
            final_scores = final_scores_after_removal
            final_cats = final_cats_after_removal


    # check for multi-name institution problems (final check)
    preds_to_remove = []
    if final_preds[0] == -1:
        pass
    else:
        final_department_name_ids = [[x, str(full_affiliation_dict[x]['display_name'])] for x in final_preds if
                       (str(full_affiliation_dict[x]['display_name']).startswith("Department of") |
                        str(full_affiliation_dict[x]['display_name']).startswith("Department for"))]
        if final_department_name_ids:
            for temp_id in final_department_name_ids:
                if string_match_clean(temp_id[1]) not in string_match_clean(str(raw_sentence).strip()):
                    preds_to_remove.append(temp_id[0])
                elif not check_for_city_and_country_in_string(raw_sentence, countries,
                                                              full_affiliation_dict[temp_id[0]]):
                    preds_to_remove.append(temp_id[0])
                else:
                    pass


        if any(x in final_preds for x in multi_inst_names_ids):
            # go through logic
            if len(final_preds) == 1:
                pred_name = str(full_affiliation_dict[final_preds[0]]['display_name'])
                # check if it is exact string match
                if (string_match_clean(pred_name) == string_match_clean(str(raw_sentence).strip())):
                    final_preds = [-1]
                    final_scores = [0.0]
                    final_cats = ['nothing']
                elif pred_name.startswith("Department of"):
                    if ("College" in raw_sentence) or ("University" in raw_sentence):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['nothing']
                    elif string_match_clean(pred_name) not in string_match_clean(str(raw_sentence).strip()):
                        final_preds = [-1]
                        final_scores = [0.0]
                        final_cats = ['nothing']

            else:
                non_multi_inst_name_preds = [x for x in final_preds if x not in multi_inst_names_ids]
                if len(non_multi_inst_name_preds) > 0:
                    for temp_pred, temp_score, temp_cat in zip(final_preds, final_scores, final_cats):
                        if temp_pred not in non_multi_inst_name_preds:
                            aff_dict_temp = full_affiliation_dict[temp_pred]
                            if aff_dict_temp['display_name'].startswith("Department of"):
                                if ("College" in raw_sentence) or ("University" in raw_sentence):
                                    preds_to_remove.append(temp_pred)
                                elif (string_match_clean(str(full_affiliation_dict[temp_pred]['display_name']))
                                      not in string_match_clean(str(raw_sentence).strip())):
                                    preds_to_remove.append(temp_pred)
                                else:
                                    if check_for_city_and_country_in_string(raw_sentence, countries, aff_dict_temp):
                                        pass
                                    else:
                                        preds_to_remove.append(temp_pred)
                            # check for city and country
                            elif aff_dict_temp['country'] in countries:
                                pass
                            else:
                                preds_to_remove.append(temp_pred)
                else:
                    pass
        else:
            pass

    true_final_preds = [x for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    true_final_scores = [y for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]
    true_final_cats = [z for x,y,z in zip(final_preds, final_scores, final_cats) if x not in preds_to_remove]

    if not true_final_preds:
        true_final_preds = [-1]
        true_final_scores = [0.0]
        true_final_cats = ['nothing']
    return [true_final_preds, true_final_scores, true_final_cats]

def raw_data_to_predictions(df, lang_thresh, basic_thresh):
    """
    High level function to go from a raw input dataframe to the final dataframe with affiliation
    ID prediction.
    """
    # Implementing the functions above
    df['lang'] = df['affiliation_string'].apply(get_language)
    df['country_in_string'] = df['affiliation_string'].apply(get_country_in_string)
    df['comma_split_len'] = df['affiliation_string'].apply(lambda x: len([i if i else "" for i in
                                                                          x.split(",")]))

    # Gets initial indicator of whether or not the string should go through the models
    df['affiliation_id'] = df.apply(lambda x: get_initial_pred(x.affiliation_string, x.lang,
                                                               x.country_in_string, x.comma_split_len), axis=1)

    # Filter out strings that won't go through the models
    to_predict = df[df['affiliation_id']==0.0].drop_duplicates(subset=['affiliation_string']).copy()
    to_predict['affiliation_id'] = to_predict['affiliation_id'].astype('int')

    # Decode text so only ASCII characters are used
    to_predict['decoded_text'] = to_predict['affiliation_string'].apply(unidecode)

    # Get predictions and scores for each model
    to_predict['lang_pred_score'] = get_language_model_prediction(to_predict['decoded_text'].to_list(),
                                                                  to_predict['country_in_string'].to_list())
    to_predict['basic_pred_score'] = get_basic_model_prediction(to_predict['decoded_text'].to_list(),
                                                                to_predict['country_in_string'].to_list())

    # Get the final prediction for each affiliation string
    to_predict['affiliation_id'] = to_predict.apply(lambda x:
                                                    get_final_prediction(x.basic_pred_score,
                                                                         x.lang_pred_score,
                                                                         x.country_in_string,
                                                                         x.affiliation_string,
                                                                         lang_thresh, basic_thresh), axis=1)

    # Merge predictions to original dataframe to get the same order as the data that was requested
    final_df = df[['affiliation_string']].merge(to_predict[['affiliation_string','affiliation_id']],
                                                how='left', on='affiliation_string')

#     final_df['affiliation_id'] = final_df['affiliation_id'].fillna(-1).astype('int')
    return final_df


print("Models initialized")

# %%
# !! {"metadata":{
# !!   "id": "f0cf1840"
# !! }}
"""
### Loading all gold data
"""

# %%
# !! {"metadata":{
# !!   "id": "88964fb4",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826994,
# !!     "user_tz": 180,
# !!     "elapsed": 5,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def get_preds_display_names(all_preds):
    if isinstance(all_preds, float):
        return []
    elif isinstance(all_preds[0][0], int):
        if all_preds[0][0] == -1:
            return []
        else:
            return [f"{i} - {full_affiliation_dict.get(i).get('display_name')}"
                    if full_affiliation_dict.get(i) else "-1 - None" for i in all_preds[0]]
    else:
        return []

# %%
# !! {"metadata":{
# !!   "id": "a819b4d1",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826994,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def get_labels_display_names(all_labels):
    if isinstance(all_labels, list):
        return [f"{i} - {full_affiliation_dict.get(i).get('display_name')}"
                    if full_affiliation_dict.get(i) else "-1 - None" for i in all_labels].copy()
    else:
        return []

# %%
# !! {"metadata":{
# !!   "id": "add7e269",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289826994,
# !!     "user_tz": 180,
# !!     "elapsed": 4,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def get_preds_all_or_original(all_preds, pred_type='all'):
    if isinstance(all_preds, float):
        return [-1]
    elif isinstance(all_preds[0][0], int):
        if pred_type=='all':
            return [i for i in all_preds[0]]
        else:
            final_preds = []

            for preds, scores, cats in zip(all_preds[0], all_preds[1], all_preds[2]):
                if cats == 'string_match':
                    pass
                else:
                    final_preds.append(preds)

            if not final_preds:
                final_preds = [-1]

            return final_preds
    else:
        return [-1]

# %%
# !! {"metadata":{
# !!   "id": "08921c81"
# !! }}
"""
Multiple different datasets were used for testing, refer to the documentation to find out more about them. The following code gathers all the datasets into a single dataframe so they can be run through the model.
"""

# %%
# !! {"metadata":{
# !!   "id": "b6ad63b4",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827406,
# !!     "user_tz": 180,
# !!     "elapsed": 415,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
multi_string = pd.read_csv(f"{base_path}Crudos/Datos de testeo/multi_string_inst_openalex.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
#print(multi_string.head(1))
multi_string['labels'] = multi_string['labels'].apply(lambda x: [int(i) for i in x.split("||||")])

cwts_1 = pd.read_csv(f"{base_path}Crudos/Datos de testeo/cwts_related_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
cwts_1['paper_id'] = cwts_1['paper_id'].apply(lambda x: int(x[1:]))
cwts_1['labels'] = cwts_1['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])

cwts_2 = pd.read_csv(f"{base_path}Crudos/Datos de testeo/cwts_no_relation_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
cwts_2['paper_id'] = cwts_2['paper_id'].apply(lambda x: int(x[1:]))
cwts_2['labels'] = cwts_2['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])

sampled_200 = pd.read_csv(f"{base_path}Crudos/Datos de testeo/sampled_200_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
sampled_200['labels'] = sampled_200['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])
sampled_200['dataset'] = "gold_random"

all_gold = pd.read_csv(f"{base_path}Crudos/Datos de testeo/gold_data_institution_parsing_labeled.tsv", sep="\t") \
    [['paper_id','affiliation_string','labels','dataset']]
all_gold['dataset'] = all_gold['dataset'].replace('gold_500','gold_hard').replace('gold_1000','gold_easy')
all_gold['labels'] = all_gold['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])



# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 0
# !!   },
# !!   "id": "0f1dea9a",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827407,
# !!     "user_tz": 180,
# !!     "elapsed": 39,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "8bcd2828-6dca-4b6d-c15d-27432aa4e61b"
# !! }}
##opcion 1: con datos de aws
"""all_data = pd.concat([multi_string, cwts_1, cwts_2, sampled_200, all_gold], axis=0) \
    .drop_duplicates(subset=['affiliation_string'])
all_data.head()
#all_data['labels'] = all_data['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])
all_data=all_data.explode('labels')
all_data['labels'] = all_data['labels'].astype(int)
print(all_data.info())
all_data.head(3)
all_data=all_data[
         (all_data['labels']==3130805194)|
         (all_data['labels']==4210089990)|
         (all_data['labels']==9340077)|
         (all_data['labels']==232641801)|
         (all_data['labels']==15366983)|
         (all_data['labels']==4210122565)]
print(all_data.head(2))"""


# %%
# !! {"metadata":{
# !!   "id": "OaObqGDBcoWl",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827408,
# !!     "user_tz": 180,
# !!     "elapsed": 38,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}


# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 0
# !!   },
# !!   "id": "wAltXEIzlM-M",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827413,
# !!     "user_tz": 180,
# !!     "elapsed": 43,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   },
# !!   "outputId": "b3f41c1f-19ae-4067-e3a2-bd0fb872f585"
# !! }}
### opción 2: con muestra aleatoria creada a partir de openalex en notebook 1 (spark)
all_data = pd.read_csv(f"{base_path}V2/002_Model/affiliations_para_testeo.csv")
all_data=all_data.rename(columns={"paper_ids": "paper_id"})
all_data=all_data.explode('labels','affiliation_string')
all_data['labels'] = all_data['labels'].astype(int)
all_data['dataset']="muestra_authorship"
all_data.info()
all_data.shape

#all_data=all_data[all_data['affiliation_string'].str.contains('Arg')]
all_data.head()
#all_data.value_counts()

# %%
# !! {"metadata":{
# !!   "colab": {
# !!     "base_uri": "https://localhost:8080/",
# !!     "height": 0
# !!   },
# !!   "executionInfo": {
# !!     "elapsed": 39,
# !!     "status": "ok",
# !!     "timestamp": 1732289827414,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     },
# !!     "user_tz": 180
# !!   },
# !!   "id": "Kp-vH2izlJ8g",
# !!   "outputId": "326e96a7-f417-48aa-e1a8-eacf795547ad"
# !! }}
### Opción 3: con dataset creado para argentina (Santiago)
"""all_data = pd.read_csv(f"{base_path}V2/testeo_ar.tsv",sep="\t")
all_data=all_data.explode('labels')
all_data['labels'] = all_data['labels'].str.replace('[', '')
all_data['labels'] = all_data['labels'].str.replace(']', '')
all_data['labels'] = all_data['labels'].astype(str)

all_data['labels'] = all_data['labels'].astype(int)
print(all_data.info())
print(all_data.head(3))
all_data.shape"""


# %%
# !! {"metadata":{
# !!   "id": "e971fabc"
# !! }}
"""
## Testing
"""

# %%
# !! {"metadata":{
# !!   "id": "53824b0b",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827414,
# !!     "user_tz": 180,
# !!     "elapsed": 37,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def get_confusion_matrix(labels, preds):
    TP=0
    FP=0
    TN=0
    FN=0
    if labels[0] == -1:
        if preds[0] != -1:
            FP = len(preds)
        else:
            TN = 1
    elif preds[0] == -1:
        FN = len(labels)
    else:
        TP = sum([1 for x in preds if x in labels])
        FP = sum([1 for x in preds if x not in labels])
        FN = sum([1 for x in labels if x not in preds])

    return [TP, FP, TN, FN]

# %%
# !! {"metadata":{
# !!   "id": "032677e6",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827414,
# !!     "user_tz": 180,
# !!     "elapsed": 37,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def create_preview(aff_obj):
    basic_obj = aff_obj[0]
    lang_obj = aff_obj[1]

    basic_preds = [str(x) for x in basic_obj[0]]
    basic_scores = [round(x, 5) for x in basic_obj[1]]

    lang_preds = [str(x) for x in lang_obj[0]]
    lang_scores = [round(x, 5) for x in lang_obj[1]]

    basic_affs = [institutions.get(int(x)) for x in basic_preds]

    basic_ror = [x.get('ror_id') if x else "" for x in basic_affs]
    basic_aff_name = [x.get('display_name') if x else "" for x in basic_affs]
    basic_city_name = [x.get('city') if x else "" for x in basic_affs]
    basic_country_name = [x.get('country') if x else "" for x in basic_affs]

    lang_affs = [institutions.get(int(x)) for x in lang_preds]

    lang_ror = [x.get('ror_id') if x else "" for x in lang_affs]
    lang_aff_name = [x.get('display_name') if x else "" for x in lang_affs]
    lang_city_name = [x.get('city') if x else "" for x in lang_affs]
    lang_country_name = [x.get('country') if x else "" for x in lang_affs]

    preview_df = pd.DataFrame(zip(basic_preds, lang_preds, basic_ror, lang_ror,
                                  basic_aff_name, basic_city_name, basic_country_name,
                                  basic_scores,lang_aff_name, lang_city_name,
                                  lang_country_name, lang_scores),
                             columns=['basic_pred','lang_pred','basic_ror','lang_ror','basic_aff_name',
                                      'basic_city_name', 'basic_country_name', 'basic_score','lang_aff_name',
                                      'lang_city_name','lang_country_name', 'lang_score'])

    return preview_df

# %%
# !! {"metadata":{
# !!   "id": "ceLWH-3N4L8v",
# !!   "executionInfo": {
# !!     "status": "ok",
# !!     "timestamp": 1732289827415,
# !!     "user_tz": 180,
# !!     "elapsed": 37,
# !!     "user": {
# !!       "displayName": "Maria Fernanda Artola",
# !!       "userId": "06366185027922693499"
# !!     }
# !!   }
# !! }}
def get_similar_preds_to_remove(decoded_string, curr_preds):
    """
    Looks for organizations with similar/matching names and only predicts for one of those organizations.
    """
    preds_to_remove = []
    pred_display_names = [full_affiliation_dict[i]['display_name'] for i in curr_preds]

    # Convert list to tuple or string if necessary
    # Ensure all elements are strings to create a homogeneous array
    pred_display_names = [str(x) for x in pred_display_names]

    counts_of_preds = Counter(pred_display_names)

    preds_array = np.array(curr_preds)
    preds_names_array = np.array(pred_display_names)

    for pred_name in counts_of_preds.items():
        temp_preds_to_remove = []
        to_use = []
        if pred_name[1] > 1:
            list_to_check = preds_array[preds_names_array == pred_name[0]].tolist()
            for pred in list_to_check:
                if string_match_clean(full_affiliation_dict[pred]['city']) in decoded_string:
                    to_use.append(pred)
                else:
                    temp_preds_to_remove.append(pred)
            if not to_use:
                to_use = temp_preds_to_remove[0]
                preds_to_remove += temp_preds_to_remove[1:]
            else:
                preds_to_remove += temp_preds_to_remove
        else:
            pass

    return preds_to_remove

# %%
# !! {"metadata":{
# !!   "id": "20IQ_gXNhrFK"
# !! }}
raw_data_to_predictions(all_data, lang_thresh=0.99, basic_thresh=0.99).head(4)

# %%
# !! {"metadata":{
# !!   "id": "bW7N8Qrdyuas"
# !! }}
#<cc-ac> %%time
all_preds = raw_data_to_predictions(all_data, lang_thresh=0.99, basic_thresh=0.99)\
    .merge(all_data[['paper_id','affiliation_string','labels','dataset']])

all_preds1=all_preds

# %%
# !! {"metadata":{
# !!   "id": "kkWFKC_H0cJD"
# !! }}
all_preds1.head(2)

# %%
# !! {"metadata":{
# !!   "id": "P4bauLcvzALt"
# !! }}
all_preds1['preds_name'] = all_preds1['affiliation_id'].apply(lambda x: get_preds_display_names(x))

all_preds1['preds_model_and_string_matching'] = all_preds1['affiliation_id'] \
    .apply(lambda x: get_preds_all_or_original(x, 'all'))
all_preds1['preds_model_only'] = all_preds1['affiliation_id']\
    .apply(lambda x: get_preds_all_or_original(x, 'model_only'))

all_preds1['preds_model_and_string_matching'] = all_preds1['preds_model_and_string_matching'] \
    .apply(lambda x: [int(i) if ~np.isnan(i) else -1 for i in x])
all_preds1['preds_model_only'] = all_preds1['preds_model_only']\
    .apply(lambda x: [int(i) if ~np.isnan(i) else -1 for i in x])

all_preds1.head(2)

# %%
# !! {"metadata":{
# !!   "id": "gycTKbwq_pVY"
# !! }}
all_preds1.info()
all_preds1['labels']=all_preds['labels'].astype(int)
all_preds1.info()

# %%
# !! {"metadata":{
# !!   "id": "gxCVg7MsCbwY"
# !! }}


# %%
# !! {"metadata":{
# !!   "id": "0-hQstO0Ca5O"
# !! }}
all_preds1['labels'].head(2)

# %%
# !! {"metadata":{
# !!   "id": "jRakUBTX9o-R"
# !! }}
def get_labels_display_names(all_labels):
    if isinstance(all_labels, list):
        return [f"{i} - {full_affiliation_dict.get(i).get('display_name')}"
                    if full_affiliation_dict.get(i) else "-1 - None" for i in all_labels].copy()
    else:
        return []

get_labels_display_names(all_preds1['labels'])


# %%
# !! {"metadata":{
# !!   "id": "t-8eyFDuJTMb"
# !! }}
def get_confusion_matrix(labels, preds):
    """
    Calculates the confusion matrix for a single row of predictions.

    Args:
        labels: The true labels for the affiliation. This should be an iterable (list or array-like).
        preds: The predicted labels for the affiliation. This should be an iterable (list or array-like).

    Returns:
        A dictionary containing the confusion matrix values (TP, FP, TN, FN).
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Check if labels is a single value and convert to a list if needed
    if not isinstance(labels, (list, tuple, np.ndarray)):  # Check if it's not already iterable
        labels = [labels]  # Wrap single integer in a list
    if not isinstance(preds, (list, tuple, np.ndarray)):  # Check if it's not already iterable
        preds = [preds]  # Wrap single integer in a list


    if labels[0] == -1:
        if preds[0] != -1:
            FP = len(preds)
        else:
            TN = 1
    else:
        # Check for true positives: label is not -1 and prediction matches any label
        TP = sum(1 for pred in preds if pred in labels and pred != -1)
        # Check for false positives: prediction is not -1 and not in true labels
        FP = sum(1 for pred in preds if pred != -1 and pred not in labels)
        FN = len(labels) - TP  # False negatives: true labels missed

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

# %%
# !! {"metadata":{
# !!   "id": "VZMKNAAEGZUR"
# !! }}
##paso full_affiliation_dict a dataframe para matchear label de all_preds con display name de full_affiliation_dict
a=pd.DataFrame.from_dict(full_affiliation_dict,orient='index')
a=a[['display_name']]
a.index.name = 'labels'
print(a.head(50))
a.shape


# %%
# !! {"metadata":{
# !!   "id": "f9TyWJmkAtU0"
# !! }}

a=all_preds.merge(a,on='labels').head()
all_preds['labels_name'] = a['display_name']
all_preds.head(2)



# %%
# !! {"metadata":{
# !!   "id": "Heu-1SJL9fEz"
# !! }}
all_preds['conf_mat_model_and_string_matching'] = all_preds\
    .apply(lambda x: get_confusion_matrix(x.labels, x.preds_model_and_string_matching), axis=1)
all_preds['conf_mat_model_only'] = all_preds.apply(lambda x: get_confusion_matrix(x.labels,
                                                                                  x.preds_model_only), axis=1)
all_preds.head(2)

# %%
# !! {"metadata":{
# !!   "id": "8yvn60XD8Kwn"
# !! }}
all_preds['has_FP'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: 1 if x['FP'] > 0 else 0)
all_preds['has_FN'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: 1 if x['FN'] > 0 else 0)

all_preds['TP'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x['TP'])
all_preds['FP'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x['FP'])
all_preds['TN'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x['TN'])
all_preds['FN'] = all_preds['conf_mat_model_and_string_matching'].apply(lambda x: x['FN'])

all_preds.head(2)

# %%
# !! {"metadata":{
# !!   "id": "9acf8ca4"
# !! }}
"""
### Overall Performance
"""

# %%
# !! {"metadata":{
# !!   "id": "2d463de6"
# !! }}
"""
As mentioned above, the folowing code generates the precision and recall for the data. Predictions are split up between ones that are made by the model (model_only) and predictions that are added on using smart string-matching (model_and_string_matching).
"""

# %%
# !! {"metadata":{
# !!   "id": "a54d3e9b"
# !! }}
model_and_string_matching_confs_1000 = all_preds['conf_mat_model_and_string_matching'].tolist()
model_only_confs_1000 = all_preds['conf_mat_model_only'].tolist()

print("--------- MODEL WITH STRING MATCHING ---------")
# Access dictionary values using keys instead of indices
print("Precision: ", round(sum([x['TP'] for x in model_and_string_matching_confs_1000])/
            (sum([x['TP'] for x in model_and_string_matching_confs_1000]) +
             sum([x['FP'] for x in model_and_string_matching_confs_1000])), 3))

print("Recall: ", round(sum([x['TP'] for x in model_and_string_matching_confs_1000])/
            (sum([x['TP'] for x in model_and_string_matching_confs_1000]) +
             sum([x['FN'] for x in model_and_string_matching_confs_1000])), 3))
print("")
print("--------- MODEL ONLY ---------")

# Access dictionary values using keys instead of indices
print("Precision: ", round(sum([x['TP'] for x in model_only_confs_1000])/
            (sum([x['TP'] for x in model_only_confs_1000]) + sum([x['FP'] for x in model_only_confs_1000])), 3))

print("Recall: ", round(sum([x['TP'] for x in model_only_confs_1000])/
            (sum([x['TP'] for x in model_only_confs_1000]) + sum([x['FN'] for x in model_only_confs_1000])), 3))

# %%
# !! {"metadata":{
# !!   "id": "cLSbJ8fVlss6"
# !! }}
all_data.head(5)

# %%
# !! {"metadata":{
# !!   "id": "5L7AR3LKl686"
# !! }}
all_data['country_in_string'].value_counts()

# %%
# !! {"metadata":{
# !!   "id": "WdSCg5_qgzk3"
# !! }}
sub_p_res = subprocess.run(['pip', 'install', 'colab-convert'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
print(sub_p_res) #<cc-cm>
sub_p_res = subprocess.run(['colab-convert', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/003_model_testing.ipynb', '/content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/003_model_testing.p'], stdout=subprocess.PIPE).stdout.decode('utf-8') #<cc-cm>
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
