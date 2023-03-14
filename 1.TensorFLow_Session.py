import tensorflow as tf

matrix1 = tf.constant([[2, 2]])
matrix2 = tf.constant([[1],[2]])

sess = tf.Session()
product = tf.matmul(matrix1, matrix2)
result = sess.run(product)
print(result)

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)