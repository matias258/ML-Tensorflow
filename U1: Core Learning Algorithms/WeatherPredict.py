"""
Vamos a modelar un sistema climático simple e intentar predecir la temperatura de cada día, teniendo en cuenta la siguiente información:

1. Los días fríos se codifican con un 0 y los días calurosos se codifican con un 1.
2. El primer día de nuestra secuencia tiene un 80% de probabilidad de ser frío.
3. Un día frío tiene un 30% de probabilidad de ser seguido por un día caluroso.
4. Un día caluroso tiene un 20% de probabilidad de ser seguido por un día frío.
5. En cada día, la temperatura sigue una distribución normal con una media y desviación estándar de 0 y 5 en un día frío, y una media y desviación estándar de 15 y 10 en un día caluroso.
"""

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

#En este ejemplo, en un día caluroso la temperatura promedio es de 15 y varía entre 5 y 25.
#Para modelar esto en TensorFlow, haremos lo siguiente.
tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Haciendo referencia al punto 2 anterior
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],  #Si el dia es cold, 70% chance de que siga cold.
                                                 [0.2, 0.8]]) #Si el dia es hot, 80% chance de que siga hot.
                                                              # Haciendo referencia a los puntos 3 y 4 
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # Haciendo referencia al punto 5
# en dia hot, es 0 y 15 en cold day. La standard deviation es 5 en cold day, y 10 en hot day.
# the loc argument represents the mean and the scale is the standard devitation


#Ahora hemos creado variables de distribución para modelar nuestro
#sistema y es hora de crear el modelo oculto de Markov.
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)
#steps es la ctidad de dias que queremos predecir, en este caso una semana.


#Para obtener las temperaturas esperadas en cada día, podemos hacer lo siguiente.
mean = model.mean()

# Debido a la forma en que TensorFlow funciona a un nivel inferior, necesitamos evaluar parte del gráfico
# desde dentro de una sesión para ver el valor de este tensor.

# En la nueva versión de TensorFlow, debemos utilizar tf.compat.v1.Session() 
# en lugar de simplemente tf.Session() para crear una sesión.
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())   #obtengo:[11.999999 10.500001  9.75      9.375     9.1875    9.09375   9.046875]
                        #que son las expected temperatures on each day en grados C.
