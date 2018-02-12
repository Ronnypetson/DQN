import tensorflow as tf
import gym
import numpy as np
import os
import random
from collections import deque

env_name = 'CartPole-v0'
state_dim = 4
ob_frames = 1
num_keys = 2
learning_rate = 0.01
batch_size = 64
replay_len = 100000
oldest_mem = 0
default_action = 1
empty_obs = np.zeros((ob_frames,state_dim))
mem = replay_len*[{'q_sa':0.0,'obs':empty_obs,'act':default_action,'r':0.0,'new_obs':empty_obs,'d':False}]
model_fn = 'checkpoint/'+env_name+'/'+env_name+'.ckpt'

def beep():
	duration = 1  # second
	freq = 440  # Hz
	os.system('play --no-show-progress --null --channels 0.25 synth %s sine %f' % (duration, freq))

def Or(f):
	for a in f:
		if a:
			return True
	return False

def get_argmaxes(a): #
	m = []
	for i in range(batch_size):
		b = a[2*i:2*i+num_keys]
		m.append(2*i+np.argmax(b))
	return m

def replace_mem(new_):
	global oldest_mem
	global mem
	mem[oldest_mem] = new_
	oldest_mem = (oldest_mem+1)%replay_len

def get_batch():
	q_sa = []
	ob = []
	act = []
	r = []
	new_ob = []
	d = []
	for i in range(batch_size):
		reg = random.choice(mem)
		for j in range(num_keys):
			q_sa.append(reg['q_sa'])
			ob.append(reg['obs'])
			a = np.zeros(num_keys)
			a[reg['act']] = 1.0
			act.append(a)
			r.append(reg['r'])
			new_ob.append(reg['new_obs'])
			d.append(reg['d'])
	return q_sa, ob, act, r, new_ob, d

def step(env,a,render=False):
	obs = np.zeros((ob_frames,state_dim))
	r = np.zeros(ob_frames)
	d = ob_frames*[False]
	for i in range(ob_frames):
		if render: env.render()
		obs[i],r[i],d[i],_ = env.step(a)
		if d[i]:
			break
	r = np.sum(r)
	d = Or(d)
	return obs,r,d

def set_rep_mem(env):
	global mem
	obs = np.expand_dims(env.reset(),axis=0)
	for i in range(replay_len):
		a = env.action_space.sample()
		new_obs,r,d = step(env,a)
		mem[i] = {'q_sa': 0.0,'obs':obs,'act':a,'r':r,'new_obs':new_obs,'d':d}
		if d:
			obs = np.expand_dims(env.reset(),axis=0)
		else:
			obs = new_obs

X = tf.placeholder(tf.float32,[None,ob_frames,state_dim])
act = tf.placeholder(tf.float32,[None,num_keys])
Y = tf.placeholder(tf.float32,[None,1])

X_ = tf.contrib.layers.flatten(X)
act_ = tf.contrib.layers.flatten(act)
fc1 = tf.concat([X_,act_],1)
fc1 = tf.layers.dense(fc1,50,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,10,activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2,1,activation=tf.nn.relu)

loss = tf.losses.mean_squared_error(fc3,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

env = gym.make(env_name)
gamma = 0.99
e = 0.01
alpha = 1.0

single_action = np.identity(num_keys).tolist()
batch_action = batch_size*single_action

with tf.Session() as sess:
	saver = tf.train.Saver()
	if os.path.isfile(model_fn+'.meta'):
		saver.restore(sess,model_fn)
	else:
		sess.run(tf.global_variables_initializer())
	#set_rep_mem(env)
	scores = deque(maxlen=100)
	for t in range(500000):
		obs = np.expand_dims(env.reset(),axis=0)
		d = False
		i = 0
		while not d:
			allQ = sess.run(fc3,feed_dict={X:num_keys*[obs],act:single_action})
			allQ = np.transpose(allQ)[0]
			#a = np.random.choice(np.flatnonzero(allQ == allQ.max()))
			a = np.argmax(allQ)
			if np.random.rand(1) < e:
				a = env.action_space.sample()
			new_obs,r,d = step(env,a) # render=(t/500 % 8 == 7)
			new_mem = {'q_sa': allQ[a],'obs':obs,'act':a,'r':r,'new_obs':new_obs,'d':d}
			replace_mem(new_mem) #
			obs = new_obs
			i += 1
		scores.append(i)
		print(np.mean(scores))
		# Replay
		q_sa, b_ob, b_act, b_r, b_new_ob, b_d = get_batch()
		maxQ = sess.run(fc3,feed_dict={X:b_new_ob,act:batch_action}) ##
		maxQ = np.transpose(maxQ)[0]
		#
		argmaxQ = get_argmaxes(maxQ) #
		b_d = [b_d[j] for j in argmaxQ]
		b_r = [b_r[j] for j in argmaxQ]
		#q_sa = [q_sa[j] for j in argmaxQ]
		y = np.zeros(batch_size)
		for j in range(batch_size):
			if b_d[j]:
				y[j] = b_r[j]
			else:
				y[j] = b_r[j] + gamma*maxQ[argmaxQ[j]]
		s_y = np.sum(y)/batch_size
		e = 0.1/(1+np.exp(s_y/100))
		#print(s_y)
		x_ = [b_ob[j] for j in argmaxQ] #
		ac_ = [b_act[j] for j in argmaxQ] #
		#print(ac_)
		y = np.expand_dims(y,axis=1)
		sess.run(train,feed_dict={X:x_,act:ac_,Y:y})
		if t%1000 == 99:
			saver.save(sess,model_fn)

