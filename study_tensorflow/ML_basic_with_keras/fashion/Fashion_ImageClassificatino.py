# Study referecne
# https://www.tensorflow.org/tutorials/keras/classification

from cgi import test
from pickletools import optimize
from pyexpat import model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

fashoin_mnist = tf.keras.datasets.fashion_mnist
(trainImage, trainLabel), (testImage, testLabel) = fashoin_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(trainImage.shape)

# matpotlib window setting
plt.figure()

# Show First dataset of fashion MNIST
plt.imshow(trainImage[0])

# matpotlib window add params and finally rendering
plt.colorbar()
#plt.show()

trainImage = trainImage / 255.0
testImage = testImage / 255.0

'''
Category of the fashion MNIST class exmaple images on there.
So 25 classes on fashion MNIST (I'll call it f-mnist) as u can see it on plot window
'''
plt.figure(figsize=(10, 10))                            # Size of window
for i in range(25):                                     # 25 means are classes length on f-mnist. looping for 25times that can read and show all of class examples
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainImage[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[trainLabel[i]])
#plt.show()

# ================================ #

'''Build Model Part
First up, we need up the set up layers.

layers order by the code. So as you can see Flatten is the first of the models and layer.
flatten is reformating input data for better preconcious of make good model
and second layer called dense is neural layers. I set it 128 nodes (a.k.a neuros).
and last layer is same second layer but order is difference. that return final output to logits array with length of 10.

according to tensorflow docs, those node and layers that figure in out 10 classes just on this model.
'''
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(0)
])

'''Compile the model
I made up the model above there and now we compile da model that we made. 
optimizer is loss fuctions and loss is for model accurate is minimize loss fuction to "steer" the model go to right direction
metrics is for monitor the training and testing steps or images that are correctly classified'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
'''And finally train the model.
Its call fee the model. it just for the starting for trainning'''
model.fit(trainImage, trainLabel, epochs=10)

testLoss, testAcc = model.evalute(testImage, testLabel, verbose=2)
print('\nTest accuracy : ', testAcc)

'''Make predictions
model needs some notebook that can be reference for the study. this predicitions could be.'''
probabilityModel = tf.keras.Sequential([model, tf.kears.layers.Softmax()])
predicitions = probabilityModel.predict(testImage)

print("Predicited for each image : {}\nHighest Confidence Value : {}\nComparing test label shows that classification : {}".format(predicitions[0], np.argmix(predicitions[0]), testLabel[0]))

def plotImage(i, predictionArray, trueLabel, img):
    trueLabel, img = trueLabel[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicitedLabel = np.argmax(predictionArray)
    if predicitedLabel == trueLabel:
        color = 'bule'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})". format(class_names[predicitedLabel],
                                         100 * np.max(predictionArray),
                                         class_names[trueLabel]),
                                         color = color)

def plotValueArray(i, predictionsArray, trueLabel):
    trueLabel = trueLabel[i]
    plt.xticks(range(10))
    plt.yticks([])
    thisPlot = plt.bar(range(10), predictionsArray, color="#777777")
    plt.ylim([0, 1])
    predicitedLabel = np.argmax(predictionsArray)

    thisPlot[predicitedLabel].set_color('red')
    thisPlot[trueLabel].set_color('blue')

i = 0     # i means select class

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plotImage(i, predicitions[i], testLabel, testImage)

plt.subplot(1, 2, 2)
plotValueArray(i, predicitions[i], testLabel)

plt.show()