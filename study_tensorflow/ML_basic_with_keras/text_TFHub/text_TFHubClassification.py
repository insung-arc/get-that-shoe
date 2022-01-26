# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
'''Okay, this code is training text from tensorflow hub and datasets.
Binary or two-class-classification code using by Tensorflow Hub and Keras.
According to offical tensorflow docs, IMDB datasets has 50,000 movie reviews from Internet Moive Database (called IMDB).
Training and testing sets are balanced, mean equal number of positvie and negative. You may guess about what is two class
yea, thats right. Postive class and negative class. 
'''
import os
from pickletools import optimize
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Print Information of tensorflow
print("Version: \t", tf.__version__)
print("Eager mode: \t", tf.executing_eagerly())
print("Hub version: \t", hub.__version__)
print("GPU is",
    "available" if tf.config.list_physical_devices("GPU")
                else "NOT AVAILABLE")

# Load IMDB reviews from tensorflow datasets.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

train_example_batch, train_labels_batch = next(iter(train_data.batch(10)))
# this is a example of ml datasets
print(train_example_batch)
# and this a first 10 labels
print(train_labels_batch)

'''Now its very important and very cool part
BUILD THE MODEL
just read this docs -> https://www.tensorflow.org/tutorials/keras/text_classification_with_hub#build_the_model
'''
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_example_batch[:3])

# Stacking model layers!
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

'''Loss function and optimizer
Okay model need loss for import the data and training.
Like human learns by mistake, and machine learns by loss. Like this shit.'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# evaluate the model
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics, results):
    print("%s: %.3f" % (name, value))

    