# -*- coding: utf-8 -*-
"""002a_basic_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nZjHRO-NL7YxTXu7yzxImcmB4c3cW6_6
"""

import pickle
import json
import os
import math
!pip install unidecode
import unidecode
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from math import ceil
from sklearn.model_selection import train_test_split

!pip install colab-convert ## comentar en script
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
# Import TFSMLayer from tensorflow.keras.layers
from tensorflow.keras.layers import TFSMLayer
import re # Import the 're' module for regular expressions

# HuggingFace library to train a tokenizer
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

from google.colab import drive

drive.mount('/content/drive') ## comentar en script

"""**Cambiar path para correr desde otro lado**"""

base_path = '/content/drive/MyDrive/openalex-institution-parsing/'

"""### Combining the training data from 001 notebook and artificial data"""

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

full_affs_data = pd.concat([more_than, lower_than],
                           axis=0).reset_index(drop=True)

full_affs_data.info()

full_affs_data.head()

##guardamos como parquet
full_affs_data.to_parquet(f'{base_path}V2/002_Model/full_affs_data.parquet')

full_affs_data.shape

full_affs_data['text_len'] = full_affs_data['original_affiliation'].apply(len)

full_affs_data = full_affs_data[full_affs_data['text_len'] < 500][['original_affiliation','affiliation_id']].copy()

full_affs_data.shape

full_affs_data['affiliation_id'] = full_affs_data['affiliation_id'].astype('str')

full_affs_data.head(n=3)

full_affs_data.info()



"""### Processing and splitting the data"""

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

print(full_affs_data.head())

train_data, val_data = train_test_split(full_affs_data, train_size=0.80, random_state=1)
train_data = train_data.reset_index(drop=True).copy()
val_data = val_data.reset_index(drop=True).copy()

print(train_data.shape)

print(val_data.shape)

affs_list_train = train_data['processed_text'].tolist()
affs_list_val = val_data['processed_text'].tolist()
affs_list_val[:5]

try:
    os.system("rm {base_path}V2/002_Model/aff_text.txt")
    print("Done")
except:
    pass

# save the affiliation text that will be used to train a tokenizer
#with open("aff_text.txt", "w") as f:
base_path = '/content/drive/MyDrive/openalex-institution-parsing/'
with open(f"{base_path}V2/002_Model/aff_text.txt", "w") as f: # Added 'w' to open in write mode
    for aff in affs_list_train:
        f.write(f"{aff}\n")

try:
    os.system("rm {base_path}V2/002_Model/basic_model_tokenizer")
    print("Done")
except:
    pass

#full_affs_data[['processed_text','affiliation_id']].to_parquet("full_affs_data_processed.parquet")
full_affs_data[['processed_text','affiliation_id']].to_parquet(f"{base_path}V2/002_Model/full_affs_data_processed.parquet")

"""### Creating the tokenizer for the basic model"""

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

print(wordpiece_tokenizer)

"""### Further processing of data with tokenizer"""

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

# initializing an empty affiliation vocab
affiliation_vocab = {}

# tokenizing the training dataset
tokenized_output = []
for i in affs_list_train:
    tokenized_output.append(wordpiece_tokenizer.encode(i).ids)

train_data['original_affiliation_tok'] = tokenized_output
print(train_data['original_affiliation_tok'].head(1))

# tokenizing the validation dataset
tokenized_output = []
for i in affs_list_val:
    tokenized_output.append(wordpiece_tokenizer.encode(i).ids)

val_data['original_affiliation_tok'] = tokenized_output

# applying max length cutoff and padding
train_data['original_affiliation_model_input'] = train_data['original_affiliation_tok'].apply(max_len_and_pad)
val_data['original_affiliation_model_input'] = val_data['original_affiliation_tok'].apply(max_len_and_pad)

train_data['original_affiliation_tok'].apply(max_len_and_pad).head(1)

val_data['original_affiliation_tok'].apply(max_len_and_pad).head(1)

# creating the label affiliation vocab
train_data['label'] = train_data['affiliation_id'].apply(lambda x: create_affiliation_vocab(x))

train_data['affiliation_id'].apply(lambda x: create_affiliation_vocab(x)).head(3)

print(train_data.shape)

print(len(affiliation_vocab))

val_data['label'] = val_data['affiliation_id'].apply(lambda x: [affiliation_vocab.get(x)])

val_data['affiliation_id'].apply(lambda x: [affiliation_vocab.get(x)]).head(3)

print(val_data.shape)

dict(list(affiliation_vocab.items())[0:10])



train_data.to_parquet(f"{base_path}V2/002_Model/training_data/train_data.parquet")
val_data.to_parquet(f"{base_path}V2/002_Model/training_data/val_data.parquet")

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

print(len(affiliation_vocab))

"""### Creating TFRecords from the training and validation datasets"""

train_data = pd.read_parquet(f"{base_path}V2/002_Model/training_data/train_data.parquet")

val_data = pd.read_parquet(f"{base_path}V2/002_Model/training_data/val_data.parquet")

# saving the affiliation vocab
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","rb") as f:
    affiliation_vocab = pickle.load(f)

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

def tf_serialize_example(f0, f1):
    """
    Serialization function.
    """
    tf_string = tf.py_function(serialize_example, (f0, f1), tf.string)
    return tf.reshape(tf_string, ())

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

# Making sure data is in the correct format before going into TFRecord
train_data['original_affiliation_model_input'] = train_data['original_affiliation_model_input'] \
.apply(lambda x: np.asarray(x, dtype=np.int64))

val_data['original_affiliation_model_input'] = val_data['original_affiliation_model_input'] \
.apply(lambda x: np.asarray(x, dtype=np.int64))

os.system(f"mkdir -p {base_path}V2/002_Model/training_data/train/")
os.system(f"mkdir -p {base_path}V2/002_Model/training_data/val/")
print("Done")

"""#### Creating the Train Dataset"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# for i in range(ceil(train_data.shape[0]/500000)):
#     print(i)
#     low = i*500000
#     high = (i+1)*500000
#     create_tfrecords_dataset(train_data.iloc[low:high,:], i, 'train')

"""#### Creating the Validation Dataset"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# for i in range(ceil(val_data.shape[0]/60000)):
#     print(i)
#     low = i*60000
#     high = (i+1)*60000
#     create_tfrecords_dataset(val_data.iloc[low:high,:], i, 'val')

print(train_data.head(n=8))

print(train_data.info())

print(val_data.head(n=3))

print(val_data.info())

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

"""### Loading the Data"""

train_data_path = f"{base_path}V2/002_Model/training_data/"
AUTO = tf.data.experimental.AUTOTUNE
training_data = get_dataset(train_data_path, data_type='train')
validation_data = get_dataset(train_data_path, data_type='val')

print(val_data.isna().sum())

train_data.isna().sum()

"""### Load Vocab"""

# Loading the affiliation (target) vocab
with open(f"{base_path}Crudos/institution_tagger_v2_artifacts/affiliation_vocab_argentina.pkl","rb") as f:
    affiliation_vocab_id = pickle.load(f)

affiliation_vocab_id = {int(i):int(j) for i,j in affiliation_vocab_id.items()}
print(affiliation_vocab_id)
inverse_affiliation_vocab_id = {(i):(j) for j,i in affiliation_vocab_id.items()}
print(inverse_affiliation_vocab_id)

"""### Creating Model"""

# Hyperparameters to tune
emb_size = 256
max_len = 128
num_layers = 3  ## se cambió de 6 a 3
num_heads = 8
dense_1 = 2048
dense_2 = 1024
learn_rate = 0.01 ## la cambié de 0.0004 a 0.001

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

print(model.summary())

"""### Training the Model

The ***batch size*** in iterative gradient descent is the number of patterns shown to the network before the weights are updated. It is also an optimization in the training of the network, defining how many patterns to read at a time and keep in memory.

The ***number of epochs*** is the number of times the entire training dataset is shown to the network during training. Some networks are sensitive to the batch size, such as LSTM recurrent neural networks and Convolutional Neural Networks.
"""

history = model.fit(training_data, epochs=20, validation_data=validation_data, verbose=1, callbacks=callbacks)

print(model.summary())

json.dump(str(history.history), open(f"{filepath_1}_25EPOCHS_HISTORY.json", 'w+'))

model.save(f'{base_path}V2/002_Model/Result_basic_model/modelo_basico_keras.keras')
# %%
print('FINALIZADO OK')

model.export(f'{base_path}V2/002_Model/Result_basic_model/modelo_basico_export')

# para guardar archivo del modelo (codigo original no genera los archivos que se levantan en la siguiente notebook)

# Definir el directorio en la carpeta actual donde deseas guardar el modelo
saved_model_dir = f'{base_path}V2/002_Model/Result_basic_model/basic_model/'

# Crear el directorio si no existe
os.makedirs(saved_model_dir, exist_ok=True)

# Guardar el modelo en formato SavedModel usando tf.saved_model.save()
tf.saved_model.save(model, saved_model_dir)
print(f"Modelo guardado en {saved_model_dir}")

## opciones para guardar el modelo
##https://medium.com/swlh/saving-and-loading-of-keras-sequential-and-functional-models-73ce704561f4
#https://www.architecture-performance.fr/ap_blog/saving-a-tf-keras-model-with-data-normalization/

ax = pd.DataFrame(data=history.history).plot(figsize=(15, 7))
ax.grid()
_ = ax.set(title="Training loss and accuracy", xlabel="Epochs")
_ = ax.legend(["Training loss", "Trainig accuracy"])

training_data



###########################
### Probando el predict ###
###########################

# Extract the relevant data
a = val_data.iloc[1:10, 4].tolist()
basic_tok_tensor = tf.convert_to_tensor(a, dtype=tf.int64)
# Make predictions
tf.math.top_k(model.predict(basic_tok_tensor),2)

#Lo que da la label es la posición que tiene determinada institución en el diccionario.

###con dataset creado para argentina (Santiago)
all_data = pd.read_csv(f'{base_path}V2/testeo_ar.tsv',sep="\t")
all_data['labels'] = all_data['labels'].apply(lambda x: [int(i.strip()) for i in x[1:-1].split(",")])
all_data=all_data.explode('labels')
all_data['labels'] = all_data['labels'].astype(int)
print(all_data.info())
all_data.head(3)
all_data.shape

to_predict=all_data
# Decode text so only ASCII characters are used


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

basic_model_path=f'{base_path}V2/002_Model/Result_basic_model/basic_model/'
# Instead of using tf.keras.models.load_model, use TFSMLayer to load the SavedModel
basic_model = TFSMLayer(basic_model_path, call_endpoint='serving_default')

# Convert the padded data to a TensorFlow tensor
basic_tok_data=decoded_text['basic_tok_data'].tolist()
basic_tok_tensor = tf.convert_to_tensor(basic_tok_data, dtype=tf.int64)

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

##comentar en python
!colab-convert /content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/002a_basic_model.ipynb /content/drive/MyDrive/openalex-institution-parsing/V2/002_Model/Notebooks/002a_basic_model.py