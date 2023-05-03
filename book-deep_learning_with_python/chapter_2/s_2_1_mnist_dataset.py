from keras.datasets import mnist
import keras
from keras import layers

# listing 2.1 Loading the MNIST dataset in Keras

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(f'train_images.shape = {train_images.shape}')
print(f'len(train_labels) = {len(train_labels)}')
print(f'train_labels = {train_labels}')
print('train_labels = ', train_labels)

# listing 2.2 the network architecture
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# listing 2.3 the compilation step
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
# listing 2.4 preparing the image data
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255  # TODO: 2023-05-03, why need to divide by 255
test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32')/255  # TODO: 2023-05-03, why need to divide by 255

# listing 2.5 'fitting' the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# listing 2.6 using the model to make predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print('predictions[0]:', predictions[0])

print('predictions[0].argmax():', predictions[0].argmax())
print('predictions[0][7]:', predictions[0][7])
print('test_labels[0]:', test_labels[0])

# listing 2.7 evaluating the model on new data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")