import tensorflow as tf
tf.disable_eager_execution()
print(tf.__version__)

# define the inputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# define the graph
g_mean = tf.sqrt(x*y)

# run this in a session
with tf.Session() as sess:
    res = sess.run(g_mean, feed_dict={x: 2.0, y: 8.0})
    print(res)