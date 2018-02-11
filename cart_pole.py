import tensorflow as tf
import gym
import numpy as np
import os
import random

env_name = 'CartPole-v0'
state_dim = 4
ob_frames = 2
num_keys = 2
learning_rate = 0.01
batch_size = 32
replay_len = 1024
oldest_mem = 0
model_fn = 'checkpoint/'+env_name+'/'+env_name+'.ckpt'

def beep():
	duration = 1  # second
	freq = 440  # Hz
	os.system('play --no-show-progress --null --channels 0.25 synth %s sine %f' % (duration, freq))

def Or(f):
	r = False
	for a in f:
		r = r or a
	return r

def get_argmaxes(a):
	m = []
	for i in range(batch_size):
		b = a[2*i:2*i+num_keys]
		m.append(np.random.choice(np.flatnonzero(b == b.max())))
	return m

def replace_mem(mem,new_):
	global oldest_mem
	mem[oldest_mem] = new_
	oldest_mem = (oldest_mem+1)%replay_len

def get_batch(m):
	ob = []
	act = []
	r = []
	new_ob = []
	d = []
	for i in range(batch_size):
		reg = random.choice(m)
		for j in range(num_keys):
			ob.append(reg['obs'])
			a = np.zeros(num_keys)
			a[reg['act']] = 1.0
			act.append(a)
			r.append(reg['r'])
			new_ob.append(reg['new_obs'])
			d.append(reg['d'])
	return ob, act, r, new_ob, d

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

single_action = np.identity(num_keys).tolist()
batch_action = batch_size*single_action
empty_obs = np.zeros((ob_frames,state_dim))
default_action = 1
replay_memory = replay_len*[{'obs':empty_obs,'act':default_action,'r':0.0,'new_obs':empty_obs,'d':False}]

with tf.Session() as sess:
	saver = tf.train.Saver()
	if os.path.isfile(model_fn+'.meta'):
		saver.restore(sess,model_fn)
	else:
		sess.run(tf.global_variables_initializer())
	for t in range(500000):
		env.reset()
		obs,r,d = step(env,default_action)
		for i in range(2000000):
			allQ = sess.run(fc2,feed_dict={X:num_keys*[obs],act:single_action})
			allQ = np.transpose(allQ)[0]
			a = np.random.choice(np.flatnonzero(allQ == allQ.max()))
			if np.random.rand(1) < e:
				a = env.action_space.sample()
			new_obs,r,d = step(env,a)
			new_mem = {'obs':obs,'act':a,'r':r,'new_obs':new_obs,'d':d}
			replace_mem(replay_memory, new_mem)
			#
			b_ob, b_act, b_r, b_new_ob, b_d = get_batch(replay_memory)
			maxQ = sess.run(fc2,feed_dict={X:b_ob,act:batch_action})
			maxQ = np.transpose(maxQ)[0]
			#
			argmaxQ = get_argmaxes(maxQ)
			b_d = [b_d[j] for j in argmaxQ]
			b_r = [b_r[j] for j in argmaxQ]
			y = np.zeros(batch_size)
			#y = np.array([b_r[k] for k in argmaxQ])+gamma*np.array([maxQ[i] for i in argmaxQ]) #
			for j in range(batch_size):
				if b_d[j]:
					y[j] = b_r[j]
				else:
					y[j] = b_r[j] + gamma*maxQ[argmaxQ[j]]
			s_y = np.sum(y)/batch_size
			e = 0.2/(1+np.exp(s_y/100))
			print(e,s_y)
			x_ = [b_ob[i] for i in argmaxQ]
			ac_ = [b_act[i] for i in argmaxQ]
			sess.run(train,feed_dict={X:x_,act:ac_,Y:np.expand_dims(y,axis=1)})
			obs = new_obs
			if d:
				break
		if t%100 == 99:
			saver.save(sess,model_fn)

