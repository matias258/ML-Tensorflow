import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

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


# Now we can pop the species column off and use that as our label.
train_y = train.pop("Species")
test_y = test.pop("Species")


# Define the feature columns
feature_columns = []
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


# Define the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(30, activation='relu', input_shape=[4]),  # first hidden layer
  tf.keras.layers.Dense(10, activation='relu'),  # second hidden layer
  tf.keras.layers.Dense(3)  # output layer
])


# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# Train the model
model.fit(train, train_y, epochs=100)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test, test_y)
print(f'Test accuracy: {test_accuracy}')
