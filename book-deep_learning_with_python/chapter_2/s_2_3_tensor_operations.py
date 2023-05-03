import time
import numpy as np
# import keras

# keras.layers.Dense(512, activation="relu")
# output = relu(dot(input, W)+ b)

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print('x.shape: ',x.shape)

x = x.reshape((6, 1))
print(x)

x = x.reshape((2, 3))
print(x)

# 2.3.1 Element-wise operations


def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x+y
    z = np.maximum(z, 0.)
print("Took: {0:.2f} s".format(time.time()-t0))

t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time()-t0))


# 2.3.2 Broadcasting
X = np.random.random((32, 10))
y = np.random.random((10,))
print('y:', y)
y = np.expand_dims(y, axis=0)
print('after expand_dims, y:', y)
Y = np.concatenate([y]*32, axis=0)
print('after concatenate, Y from y:', Y)


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)
print('z:', z)

# 2.3.3 Tensor product
x = np.random.random((32,))
y = np.random.random((32,))
z = np.dot(x, y)
print('x:', x, '\ny:', y)
print('after np.dot, z:', z)


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i]*y[i]
    return z
