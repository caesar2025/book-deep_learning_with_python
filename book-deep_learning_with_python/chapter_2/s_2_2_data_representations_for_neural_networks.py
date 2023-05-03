

from keras.datasets import mnist
import numpy as np
x = np.array(12)
print('x', x)

x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]
              ])
print('x.ndim', x.ndim)

(train_images, train_labels), (test_images, test_lables) = mnist.load_data()
print('train_images.ndim:', train_images.ndim)
print('train_images.shape:',train_images.shape)
print('train_images.dtype:',train_images.dtype)

# listing 2.8 displaying the fourth digit
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()