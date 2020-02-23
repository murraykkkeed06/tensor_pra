import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def layer(input_dim,output_dim,inputs,activation=None):
    w = tf.Variable(tf.random_normal([input_dim,output_dim]))
    b = tf.Variable(tf.random_normal([1,output_dim]))
    XWB = tf.matmul(inputs, w) + b
    if activation == None:
        outputs = XWB
    else:
        outputs = activation(XWB)

    return outputs


x = tf.placeholder("float",[None,4])

h = layer(4,3,x,tf.nn.relu)

y = layer(3,2,h)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    input_array = np.array([[1.,2.,3.,4.]])
    print(sess.run(y,feed_dict={x:input_array}))
    