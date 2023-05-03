import tensorflow as tf

# listing 3.10 using the GradientTape
input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
print(f'''
input_var: {input_var}
result: {result}
gradient: {gradient}
''')

# listing 3.11 Using GradientTape with constant tensor inputs
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
print(f'''
input_const: {input_const}
result: {result}
gradient: {gradient}
''')

# listing 3.12 Using nested gradient tapes to compute second-order gradients
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9*time**2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
print(f'''listing 3.12

time: {time}
position: {position}
speed: {speed}
acceleration: {acceleration}
''')