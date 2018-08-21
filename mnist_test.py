import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
learning_rate = 0.001
batch_size = 64

def neg_cross_entropy(y,y_):
	flat_y = tf.contrib.layers.flatten(y_)
	extended_y = tf.pad(y,[[0,0],[0,tf.shape(flat_y)[1]-tf.shape(y)[1]]])
	#return tf.reduce_mean(tf.reduce_sum(extended_y*tf.log(flat_y),reduction_indices=[1]))
	return tf.losses.softmax_cross_entropy(extended_y,flat_y)

X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,10])
X_ = tf.reshape(X,[-1,28,28,1])

layers = {}
with tf.variable_scope("conv1"):
	layers["conv1"] = tf.layers.conv2d(X_,16,5,strides=2,padding='same',activation=tf.nn.relu)
with tf.variable_scope("conv2"):
	layers["conv2"] = tf.layers.conv2d(layers["conv1"],32,5,strides=2,padding='same',activation=tf.nn.relu)
with tf.variable_scope("fc1"):
	layers["fc1"] = tf.layers.dense(tf.contrib.layers.flatten(layers["conv2"]),100,activation=tf.nn.relu)
with tf.variable_scope("fc2"):
	layers["fc2"] = tf.layers.dense(layers["fc1"],10,activation=tf.nn.relu)

losses = { layer_name:neg_cross_entropy(Y,layers[layer_name]) for layer_name in layers.keys() }
trainers = { layer_name:tf.train.AdamOptimizer(learning_rate=learning_rate)
					.minimize(losses[layer_name]) for layer_name in layers.keys() } # ,var_list=tf.trainable_variables(layer_name)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(layers["fc2"],1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		for l in ["fc2"]: #["conv1","conv2","fc1","fc2"]: #["fc2"]
			loss,_ = sess.run([losses[l],trainers[l]],feed_dict={X:batch_x,Y:batch_y})
			print("Training loss: %f"%loss)
	print("Test accuracy: %f"%sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels}))

