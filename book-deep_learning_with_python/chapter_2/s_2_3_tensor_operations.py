import numpy as np
# import keras

# keras.layers.Dense(512, activation="relu")
# output = relu(dot(input, W)+ b)

x = np.array([[0.,1.],
              [2.,3.],
              [4.,5.]])
print(x.shape)

x = x.reshape((6,1))
print(x)

x = x.reshape((2,3))
print(x)