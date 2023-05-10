"""
We will model a simple weather system and try to predict the temperature on each day given 
the following information.

1. Cold days are encoded by a 0 and hot days are encoded by a 1.
2. The first day in our sequence has an 80% chance of being cold.
3. A cold day has a 30% chance of being followed by a hot day.
4. A hot day has a 20% chance of being followed by a cold day.
5. On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and 
   mean and standard deviation 15 and 10 on a hot day.
"""

import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

#In this example, on a hot day the average temperature is 15 and ranges from 5 to 25.
#To model this in TensorFlow we will do the following.
tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],  #Si el dia es cold, 70% chance de que siga cold.
                                                 [0.2, 0.8]]) #Si el dia es hot, 80% chance de que siga hot.
                                                              #refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above
# en dia hot, es 0 y 15 en cold day. La standard deviation es 5 en cold day, y 10 en hot day.
# the loc argument represents the mean and the scale is the standard devitation


#We've now created distribution variables to model our 
#system and it's time to create the hidden markov model.
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)
#steps es la ctidad de dias que queremos predecir, en este caso una semana.


#To get the expected temperatures on each day we can do the following.
mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
  print(mean.numpy())   #obtengo:[11.999999 10.500001  9.75      9.375     9.1875    9.09375   9.046875]
                        #que son las expected temperatures on each day en grados C.