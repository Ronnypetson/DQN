import tensorflow as tf
import gym
import numpy as np
import os

env_name = 'MountainCar-v0'
state_dim = 2
ob_frames = 4
num_keys = 3
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
		if d[i]:
			break
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
	for t in range(5000):
		env.reset()
		obs,r,d = step(env,1)
		for i in range(2000000):
			allQ = sess.run(fc2,feed_dict={X:num_keys*[obs],act:np.identity(num_keys)})
			allQ = np.transpose(allQ)[0]
			a = np.random.choice(np.flatnonzero(allQ == allQ.max()))
			#a = np.argmax(allQ)
			if np.random.rand(1) < e:
				a = env.action_space.sample()
			new_obs,r,d = step(env,a)
			if Or([new_obs[i][0] >= 0.5 for i in range(ob_frames)]):
				r = 400.0
				beep()
			maxQ = sess.run(fc2,feed_dict={X:num_keys*[new_obs],act:np.identity(num_keys)})
			maxQ = np.transpose(maxQ)[0]
			maxQ = np.max(maxQ)
			y = (1.0-alpha)*allQ[a] + alpha*(r+gamma*maxQ)
			print(a,e,y,r)
			if d:
				y = r
			e = 0.2/(1+np.exp(y/100))
			sess.run(train,feed_dict={X:[obs],act:[np.identity(num_keys)[a]],Y:[[y]]})
			obs = new_obs
			if d:
				break
		if t%100 == 99:
			saver.save(sess,model_fn)

