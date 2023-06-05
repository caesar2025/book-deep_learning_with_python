# listing 3.2
import tensorflow as tf
import numpy as np

x = tf.random.normal(shape=(3,1), mean=0.,stddev=1.)
print('tf.random.normal: ',x)
print('np.random.normal: ',np.random.normal(size=(3,1),loc=0.,scale=1.))

x = tf.random.uniform(shape=(3,1),minval=0.,maxval=1.)
print('tf.random.uniform: ',x)
print('np.random.uniform: ',np.random.uniform(size=(3,1),low=0.,high=1.))