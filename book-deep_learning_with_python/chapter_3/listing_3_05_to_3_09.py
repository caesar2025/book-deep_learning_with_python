import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

# listing 3.5

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v)

# listing 3.6

v.assign(tf.ones((3, 1)))
print(v)

# # listing 3.7 # FIXME: ERROR-info: Cannot assign a device for operation ResourceStridedSliceAssign
# print(v[0,0])
# v[0, 0].assign(3.)
# print(v)

# listing 3.8
v.assign_add(tf.ones((3, 1)))
print("listing 3.8: using assign_add()")
print(v)

# listing 3.9
a = tf.ones((2, 2))
b = tf.square(a)
c = tf.sqrt(a)
d = b+c
e = tf.matmul(a, b) # TODO: matrix multiplication
print(f'''a:
{a}
b:
{b}
c:
{c}
d:
{d}
e:
{e}
''')
e *= d
print('e:', e)
