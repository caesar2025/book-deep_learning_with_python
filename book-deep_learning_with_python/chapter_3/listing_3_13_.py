import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# listing 3.13 Generating two classes of random points in a 2D plane
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(  # DONE: np.random.multivariate_normal
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
print(f'''listing 3.13

negative_samples: {negative_samples}
positive_samples: {positive_samples}
''')

# listing 3.14 stacking the two classes into an aray with shape (2000,2)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)  # DONE: np.vstack
print(f'''listing 3.14

inputs: {inputs}
''')

# listing 3.15
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),  # DONE: np.zeros
                    np.ones((num_samples_per_class, 1), dtype="float32")))  # DONE: np.ones
print(f'''listing 3.15

targets: {targets}
''')

# listing 3.16
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
# plt.show()

# listing 3.17
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))  # DONE: tf.random.uniform
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))                    # DONE: tf.zeros

# listing 3.18


def model(inputs):
    return tf.matmul(inputs, W)+b       # DONE: tf.matmul

# listing 3.19 the mean sqaured error loss function


def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets-predictions)       # DONE: tf.square
    return tf.reduce_mean(per_sample_losses)       # DONE: tf.reduce_mean

# listing 3.20 the training step function


learning_rate = 0.1


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])  # DONE: tf.gradient
    W.assign_sub(grad_loss_wrt_W*learning_rate)     # DONE: tf.Variable.assign_sub
    b.assign_sub(grad_loss_wrt_b*learning_rate)
    return loss


# listing 3.21 The batch training loop
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"loss at step {step}: {loss:.4f}")

predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
# plt.show()

x = np.linspace(-1, 4, 100)    # np.linspace
y = -  W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
