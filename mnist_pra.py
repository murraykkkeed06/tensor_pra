import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np
from time import time

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def plot_image(image):
    reshape = image.reshape(28,28)
    plt.imshow(reshape,cmap='binary')
    #print(reshape)
    plt.show()

def layer(output_dim,input_dim,inputs,activation=None):
    w = tf.Variable(tf.random_normal([input_dim,output_dim]))
    b = tf.Variable(tf.random_normal([1,output_dim]))
    XWB = tf.matmul(inputs, w) + b
    if activation == None:
        outputs = XWB
    else:
        outputs = activation(XWB)

    return outputs

#plot_image(mnist.train.images[0])
#print(mnist.train.labels[0])

#batch_images_xs, batch_images_ys = mnist.next_batch(batch_size=100)

x = tf.placeholder("float",[None,784])

h1 = layer(256,784,x,tf.nn.relu)

y_predict = layer(10,256,h1)

y_label = tf.placeholder("float",[None,10])

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(y_label,1),tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

trainEpochs = 15
batchSize = 100
loss_list = []
epoch_list = []
accuracy_list = []

startTime = time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(trainEpochs):
        for i in range(int(mnist.train.num_examples/batchSize)):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer,feed_dict={x:batch_x,y_label:batch_y})
        loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images, y_label:mnist.validation.labels})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("Train Epoch:", "%2d" % (epoch+1), "Loss =", "%.2f" % (loss),"Accuracy =", "%.2f" % (acc))

duration = time() - startTime
print("Train Finished takes: ", duration)

print("Accuray: ", sess.run(accuracy,feed_dict={x:mnist.test.images, y_label:mnist.test.labels}))
