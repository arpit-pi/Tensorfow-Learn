import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(5)

c = x1*x2
print(c)

with tf.Session() as sess:
    print(sess.run(c))
