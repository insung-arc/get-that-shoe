
from pickletools import optimize
import matplotlib.pyplot as plt
import tensorflow as tf
import random
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
val_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
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

print("###### Loss fuction and optimizer ######")
'''This part is important for the engaging tensorflow model fuctional order
Loss is like a study score that negative things and optimizer is study from the loss'''
model.compile(loss      = losses.BinaryCrossentropy(from_logits=True),
              optimizer = 'adam',
              metrics   = tf.metrics.BinaryAccuracy(threshold=0.0))

print("###### Train Model ######")
'''Finally train model
study from the datasets that we fitted to method'''
epochs = 10
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)

print("###### Evaluate the model ######")
'''Evaluating is like a testbench that model performs who it works
Loss and acccuracy will return'''

loss, accuracy = model.evaluate(test_ds)
print("Loss \t:", loss)
print("Accuracy \t:", accuracy)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

'''Training and validation accuracy
On just below code is for the loss'''
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''And this code is for the accuracy'''
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

'''Export the model
Okay, the time is back. Now we have to make export model that called weights 
that I just trained. Let's try it.'''

export_model = tf.keras.Sequential([
    vectorize_layer, 
    model, 
    layers.Activation('sigmoid')
])

export_model.compile(
    loss = losses.BinaryCrossentropy(from_logits=False), optimizer = "adam", metrics = ['accuracy']
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
]

'''
examples_dir = "./archive/" + random.choice(os.listdir("./archive/"))
examples = open(examples_dir, "r")
examples = examples.read()
'''

export_model.predict(examples)