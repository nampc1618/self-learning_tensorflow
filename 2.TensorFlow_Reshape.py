# Why TensorFlow Reshape?
# 1. It allows the building of complex and multidimensional data models which in turn enables visualization of data at a deep level. 
#   This also enables easy and faster debugging of all the nodes in the neural network and resolve issues without going through code.
# 2. It provides flexibility to try out different dimensions of the data and design as many views of the data and design solutions 
#   for any AI-related issues.
# 3. It is an open-sourced platform that ideas from the community constantly improves the quality of solutions provided by this.
# 4. This reshapes option enables users to build many use cases which were not thought of so far.

# --> Tensor is an array of data element arranged like a Matrix format. It can be a single dimension matrix or multidimension matrix
#     (2D, 3D, 4D, 5D, so on...). The matrix can hold integers or floating-point numbers or texts in strings.

# Example 1: Code create a single dimension tensor
# define and list a single dimension Tensor
"""import tensorflow as tf
sm = tf.constant([34, 45, 83, 627, 3, 6, 5, 10, 36, 47]) # definr a single dimension array
# print attributes of Tensor
print(sm)
with tf.Session() as sess:
    sm_value = sess.run(sm)
    print("Elements in the single dimension tensor")
    print(sm_value)
    print("Tensor Dimensions:", sm.get_shape())"""
    
#Example 2: Code to create a two-dimension tensor.   
#Python program to define and list a two dimension Tensor
'''import tensorflow as tf
dm=tf.constant([[7,8],[10,9],[23,6],[42,19],[74,91]]) # two dimension array
# printing attributes of Tensor

with tf.Session() as tses:
    dm_value = tses.run(dm)
    print ("Elements in the two dimension tensor")
    print(dm_value)
    print ("Tensor Dimensions", dm.get_shape())'''
    
# Example 3: Code to create three dimention tensor
import tensorflow as tf
dm = tf.constant([[[[7,8],[10,11],[22, 11]],[[42, 56],[67, 34],[89, 21]],[[23, 45],[56, 90],[3, 58]]]]) #three dimension array

with tf.Session() as sess:
    dm_value = sess.run(dm)
    print("Element in the Three dimension tensor")
    print(dm_value)
    print("Tensor Dimensions", dm.get_shape())

