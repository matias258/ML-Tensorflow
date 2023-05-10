
import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on


#vamos a obtener la data de keras (un submodulo de tensorflow)
#va a guardar la file como "iris_training.csv"
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")

#lo mismo para esto
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")


#load the CSV files using pandas
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe


print("train.head----")
print(train.head(), "\n")         #me devuelve los 1ros 5 rows de train

#Vemos que todo esta escrito numericalmente, nada es un string u otra cosa, son todos Numerical data.
#Entonces, al no tener Categorical data, no hace falta tener que transformar nada a numero.
# 0 -> Setosa ; 1 -> Versicolor ; 2 -> Virginica



#Now we can pop the species column off and use that as our label.
print("Popeamos Species en train_y----")
train_y = train.pop("Species")
print("\n")

print("test.pop(´Species´)----")
test_y = test.pop("Species")
print(train.head()) # la columna de "Species" no esta mas porque la popeamos.
print("\n")

print("train_y.head()----")
print(train_y.head(), "\n")

print("shape de train----")
print(train.shape, "\n")  # me devuelve (120, 4) -> tenemos 120 entries con 4 features.
#las 4 features son: SepalLength, SepalWidth, PetalLength, PetalWidth


#Input function:

# El input function es una función que toma un conjunto de datos (features y labels) 
# y los convierte en un objeto Dataset de TensorFlow. 
# Si estás en modo de entrenamiento, el conjunto de datos se mezcla y se repite. 
# Luego, se agrupa en lotes (batch) del tamaño que se indique. 
# Este objeto Dataset se utiliza posteriormente en la función de entrenamiento.


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
"""
Version deprecated:

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)
"""

# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)
print("My_feature_columns----\n",my_feature_columns, "\n")



#Buildeamos el modelo

# Build a DNN (deep neural network) with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(    #estimator almacena muchos modelos de tensorflow (DNNClassifier es uno)
    feature_columns=my_feature_columns,     #pasamos las feature_columns
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes. Porque hay 3 clases de flores, duh.
    n_classes=3)



#Una vez creado, tenemos que entrenarlo:
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
# We include a lambda to avoid creating an inner function previously

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))