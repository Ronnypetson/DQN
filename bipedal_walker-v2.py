import tensorflow as tf
import gym
import numpy as np
import os

env_name = 'BipedalWalker-v2'
state_dim = 24
ob_frames = 4
num_keys = 4
learning_rate = 0.01
model_fn = 'checkpoint/'+env_name+'/'+env_name+'.ckpt'

def beep():
	duration = 1  # second
	freq = 440  # Hz
	os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

def Or(f):
	r = False
	for a in f:
		r = r or a
	return r

def step(env,a):
	obs = np.zeros((ob_frames,state_dim))
	r = np.zeros(ob_frames)
	d = ob_frames*[False]
	for i in range(ob_frames):
		env.render()
		obs[i],r[i],d[i],_ = env.step(a)
	r = np.sum(r)
	d = Or(d)
	return obs,r,d

X = tf.placeholder(tf.float32,[None,ob_frames,state_dim])
act = tf.placeholder(tf.float32,[None,num_keys])
Y = tf.placeholder(tf.float32,[None,1])

X_ = tf.contrib.layers.flatten(X)
act_ = tf.contrib.layers.flatten(act)
fc1 = tf.concat([X_,act_],1)
fc1 = tf.layers.dense(fc1,10,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,1)

loss = tf.losses.mean_squared_error(fc2,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

env = gym.make(env_name)
gamma = 0.99
e = 0.01
alpha = 0.95

with tf.Session() as sess:
	saver = tf.train.Saver()
	if os.path.isfile(model_fn+'.meta'):
		saver.restore(sess,model_fn)
	else:
		sess.run(tf.global_variables_initializer())
	action = np.zeros((2*num_keys,num_keys))
	for i in range(2*num_keys):
		if i%2 == 0:
			action[i][i/2] = 0.5
		else:
			action[i][i/2] = -0.5
	print(action)
	for t in range(5000):
		env.reset()
		obs,r,d = step(env,num_keys*[0])
		for i in range(2000000):
			allQ = sess.run(fc2,feed_dict={X:2*num_keys*[obs],act:action})
			allQ = np.transpose(allQ)[0]
			a_ = np.random.choice(np.flatnonzero(allQ == allQ.max()))
			#a = np.argmax(allQ)
			a = action[a_]
			if np.random.rand(1) < e:
				a = env.action_space.sample()
			new_obs,r,d = step(env,a)
			#if Or([new_obs[i][0] >= 0.5 for i in range(ob_frames)]):
			#	r = 999999999999.0
			#	beep()
			maxQ = sess.run(fc2,feed_dict={X:2*num_keys*[new_obs],act:action})
			maxQ = np.transpose(maxQ)[0]
			maxQ = np.max(maxQ)
			y = (1.0-alpha)*allQ[a_] + alpha*(r+gamma*maxQ)
			print(a,e,y,r)
			if d:
				y = r
			e = 0.1/(1+np.exp(y/100))
			sess.run(train,feed_dict={X:[obs],act:[a],Y:[[y]]})
			obs = new_obs
			if d:
				break
		if t%100 == 99:
			saver.save(sess,model_fn)

