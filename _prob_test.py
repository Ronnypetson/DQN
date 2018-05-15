import tensorflow as tf
import numpy as np
import random

in_len = 5
learning_rate = 0.01
batch_size = 32
mu = [0.0,5.0,10.0,15.0,20.0]
sigma = [1.0,1.0,1.0,1.0,1.0]
n = 0
num_samples = 10000

def sample():
	global n
	n += 1 #
	return [np.random.normal(mu[i],sigma[i],1)[0] for i in range(in_len)]

def sample_uni():
	return np.random.uniform(-5.0,25.0,in_len)

def get_batch():
	return [sample() for i in range(batch_size)]

X = tf.placeholder(tf.float32, [None,in_len])
Y = tf.placeholder(tf.float32,[None,1])
fc1 = tf.layers.dense(X,20,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,1)
loss = tf.losses.mean_squared_error(fc2,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(num_samples):
		if i%2 == 1:
			s = sample()
		else:
			s = sample_uni()
		p = sess.run(fc2,feed_dict={X:[s]})[0]
		if i%2 == 1:
			p_ = p+1.0
		else:
			p_ = [0.0]
		print(s,p[0])
		l,_ = sess.run([loss,train],feed_dict={X:[s],Y:[p_]})
	#t = sample_uni()
	#t = sample()
	t = sigma
	p = sess.run(fc2,feed_dict={X:[t]})[0]
	print('Test:')
	print(t,p[0])

