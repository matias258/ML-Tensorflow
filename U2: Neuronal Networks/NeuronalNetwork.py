# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


#For this tutorial we will use the MNIST Fashion Dataset. This is a dataset that is included in keras.
#This dataset includes 60,000 images for training and 10,000 images for validation/testing.

fashion_mnist = keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training


#Let's have a look at this data to see what we are working with.
train_images.shape
#obtengo (60000, 28, 28) -> 60mil imgs que son 28 x 28, 784 pixeles en total.

train_images[0,23,23]  # let's have a look at one pixel
# -> 194 , esto significa el greyscale value del pixel

#Our pixel values are between 0 and 255, 0 being black and 255 being white. 
#This means we have a grayscale image as there are no color channels.

train_labels[:10]  # let's have a look at the first 10 training labels
# obtenemos -> array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)



# Our labels are integers ranging from 0 - 9. Each integer represents a specific article of clothing. 
# We'll create an array of label names to indicate which is which.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Fianlly let's look at what some of these images look like!
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#nos va a devolver una remera
#si ponemos train_images[0], nos devuelve una zapatilla, 3 un vestido, etc...

"""
Data Preprocessing

The last step before creating our model is to preprocess our data. 
This simply means applying some prior transformations to our data before feeding it the model. 
In this case we will simply scale all our greyscale pixel values (0-255) to be between 0 and 1. 
We can do this by dividing each value in the training and testing sets by 255.0. 
We do this because smaller values will make it easier for the model to process our values.
"""

train_images = train_images / 255.0

test_images = test_images / 255.0


# Now it's time to build the model! We are going to use a keras sequential model with three different layers. 
# This model represents a feed-forward neural network (one that passes values from left to right). 
# We'll break down each layer and its architecture below.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3), 10 porque nuestro array class_names es de 10.
# softmax se encarga de que todos las neuronas sumen hasta 1 y que sus valores esten entre 0 y 1                                                 
])

######################################################################################################################

#Compile the Model

# The last step in building the model is to define the loss function, optimizer and metrics we would like to track. 
# I won't go into detail about why we chose each of these right now.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Training the Model

# Now it's finally time to train the model. 
# Since we've already done all the work on our data this step is as easy as calling a single method.

model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!
#Luego de entrenar un tiempito, me devuelve (para 10 epochs)-> 5s 2ms/step - loss: 0.2408 - accuracy: 0.9111
#Despues de mucho boludear con el epoch, llegue a que lo mas eficiente es con 10 epochs. Osea perdi tiempo al pedo.

######################################################################################################################

# Evaluating the Model

# Now it's time to test/evaluate the model. 
# We can do this quite easily using another builtin method from keras.

# The verbose argument is defined from the keras documentation as: 
# "verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar."

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

print('Test accuracy:', test_acc)
# Generalmente el test va a dar un poco menos que lo que nos da la accuracy al entrenarlo.
# Esto se debe a que, como el modelo mira la data una y otra vez testeandola, se la empieza a memorizar.
# Esto se llama OVERFITTING!! 
# Lo que crea una ilusion de mas eficiencia, cuando no la hay.

######################################################################################################################

# Making Predictions

# To make predictions we simply need to pass an array of data in the 
# form we've specified in the input layer to .predict() method.

predictions = model.predict(test_images)
#nos da un array de (10000, 28, 28) osea un array de 10k entries de imgs  

#metamonos un poco en lo que devuelve predictions
print(predictions)          #esto nos devuelve un array con distintos valores muy chiquitos. 
                            #es la probabilidad distributiva calculada en nuestro output layer para esa img

print(np.argmax(predictions[0]))    #nos devuelve el indice del max de este array
                                    # que va a ser el que mas probabilidades tiene de salir, en este caso es el indice 9

print(class_names[np.argmax(predictions[0])])   #me devuelve la clase que es ese indice
                                                #como el indice va a ser 9, entonces la img tiene q ser la 9na (ankle boot)                                                        

#ilustremosla ya que tamos
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

######################################################################################################################


# Verifying Predictions

# small function to help us verify predictions with some simple visuals.
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
