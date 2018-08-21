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

# Input plceholders
X0 = tf.placeholder(tf.float32, [None,784])
X1 = tf.placeholder(tf.float32, [None,784])
Y0 = tf.placeholder(tf.float32, [None,10])
Y1 = tf.placeholder(tf.float32, [None,10])
X0_ = tf.reshape(X0,[-1,28,28,1])
X1_ = tf.reshape(X1,[-1,28,28,1])

# Middle placeholder
# Z = tf.placeholder(tf.float32, [None,14,14,16])

layers = {}
with tf.variable_scope("conv0_1"):
	layers["conv0_1"] = tf.layers.conv2d(X0_,16,5,strides=2,padding='same',activation=tf.nn.relu)
with tf.variable_scope("conv1_1"):
	layers["conv1_1"] = tf.layers.conv2d(X1_,16,5,strides=2,padding='same',activation=tf.nn.relu)

with tf.variable_scope("conv_shared"):
	z = tf.add(layers["conv0_1"],layers["conv1_1"])
	layers["conv_shared"] = tf.layers.conv2d(z,32,5,strides=2,padding='same',activation=tf.nn.relu)

with tf.variable_scope("fc0_1"):
	layers["fc0_1"] = tf.layers.dense(tf.contrib.layers.flatten(layers["conv_shared"]),100,activation=tf.nn.relu)
with tf.variable_scope("fc1_1"):
	layers["fc1_1"] = tf.layers.dense(tf.contrib.layers.flatten(layers["conv_shared"]),100,activation=tf.nn.relu)

with tf.variable_scope("fc0_2"):
	layers["fc0_2"] = tf.layers.dense(tf.contrib.layers.flatten(layers["fc0_1"]),10,activation=tf.nn.relu)
with tf.variable_scope("fc1_2"):
	layers["fc1_2"] = tf.layers.dense(tf.contrib.layers.flatten(layers["fc1_1"]),10,activation=tf.nn.relu)

#losses = { layer_name:neg_cross_entropy(Y,layers[layer_name]) for layer_name in layers.keys() }
#loss0 = tf.losses.softmax_cross_entropy(Y0,layers["fc0_2"])
#loss1 = tf.losses.softmax_cross_entropy(Y1,layers["fc1_2"])
loss0 = tf.losses.mean_squared_error(Y0,layers["fc0_2"])
loss1 = tf.losses.mean_squared_error(Y1,layers["fc1_2"])

###
#trainers = { layer_name:tf.train.AdamOptimizer(learning_rate=learning_rate)
#					.minimize(losses[layer_name]) for layer_name in layers.keys() } # ,var_list=tf.trainable_variables(layer_name)
vars0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv0_1")
vars0.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv_shared"))
vars0.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc0_1"))
vars0.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc0_2"))
#
vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv1_1")
vars1.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"conv_shared"))
vars1.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc1_1"))
vars1.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"fc1_2"))
#["conv0_1","conv_shared","fc0_1","fc0_2"]
#vars1 = ["conv1_1","conv_shared","fc1_1","fc1_2"]
#
trainer0 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss0,var_list=vars0)
# ,var_list=tf.trainable_variables(["conv0_1","conv_shared","fc0_1","fc0_2"])
trainer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss1,var_list=vars1)
# ,var_list=tf.trainable_variables(["conv1_1","conv_shared","fc1_1","fc1_2"]))
###

###
correct_prediction_0 = tf.equal(tf.argmax(Y0,1), tf.argmax(layers["fc0_2"],1))
accuracy_0 = tf.reduce_mean(tf.cast(correct_prediction_0, tf.float32))
#
correct_prediction_1 = tf.equal(tf.argmax(Y1,1), tf.argmax(layers["fc1_2"],1))
accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, tf.float32))
###

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(2000):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		noise_x = np.random.normal(0.0,0.01,[batch_size,784])
		noise_y = np.random.normal(0.0,0.01,[batch_size,10])
		if i%2 == 0:
			loss,_ = sess.run([loss0,trainer0],feed_dict={X0:batch_x,X1:noise_x,Y0:batch_y,Y1:noise_y})
		else:
			new_indices = np.array([(np.argmax(y_)+1)%10 for y_ in batch_y])
			batch_y = np.zeros([batch_size,10])
			batch_y[np.arange(batch_size), new_indices] = 1.0
			loss,_ = sess.run([loss1,trainer1],feed_dict={X0:noise_x,X1:batch_x,Y0:noise_y,Y1:batch_y})
		print("Training loss: %f"%loss)
	x_test = mnist.test.images[:batch_size]
	y_test_0 = mnist.test.labels[:batch_size]
	new_indices = np.array([(np.argmax(y_)+1)%10 for y_ in y_test_0])
	y_test_1 = np.zeros([batch_size,10])
	y_test_1[np.arange(batch_size), new_indices] = 1.0
	acc0, acc1 = sess.run([accuracy_0,accuracy_1], \
						feed_dict={X0:x_test,X1:x_test,Y0:y_test_0,Y1:y_test_1})
	print("Test accuracy: %f %f"%(acc0,acc1))

