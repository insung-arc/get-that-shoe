
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
import string
import re
import os

from tensorflow.keras import layers
from tensorflow.keras import losses

def custom_standradization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# download dataset from url
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

# Read sampel data
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# Load dataset 
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

print("###### Load dataset from keras dataset ######")
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size       = batch_size,
    validation_split = 0.2,
    subset           = 'training',
    seed             = seed)

for text_batch, label_batch, in raw_train_ds.take(1):
    for i in range(3):
        print("Review: \t", text_batch.numpy()[i])
        print("Label: \t", label_batch.numpy()[i])

print("Label 0 corresponds to ", raw_train_ds.class_names[0])
print("Label 1 corresponds to ", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size       = batch_size,
    validation_split = 0.2,
    subset           = 'validation',
    seed             = seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size = batch_size
)

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize             = custom_standradization, 
    max_tokens              = max_features,
    output_mode             = 'int',
    output_sequence_length  = sequence_length
)

# text-only datasets except labelds, then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review: \t", first_review)
print("Label: \t", first_label)
print("Vectorized review: \t", vectorize_text(first_review, first_label))

# these are meaning the vocabulary from the data.
# so take token each integer by .get_vocabulary() function
print("1287 ---> \t", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> \t", vectorize_layer.get_vocabulary()[313])

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

print("###### Configure the dataset for perfomance ######")
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = train_ds.cache().perfetch(buffer_size=AUTOTUNE)
test_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("###### Create the model ######")
embedding_dim = 16
model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.summary()