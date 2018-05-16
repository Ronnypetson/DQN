import tensorflow as tf
import gym
import numpy as np
import os
import random
from collections import deque

env_name = 'Pong-ram-v0'
state_dim = 128
ob_frames = 3
num_keys = 6
learning_rate = 0.001
replay_len = 10000
oldest_mem = 0
batch_size = 50
all_actions = np.identity(num_keys)
default_action = all_actions[0]
rep_all_actions = np.tile(all_actions,(batch_size,1))
empty_obs = np.zeros((ob_frames,state_dim))
model_fn = 'checkpoint/'+env_name+'/'+env_name+'.ckpt'
mem = replay_len*[{'q_sa':0.0,'obs':empty_obs,'act':default_action,'r':0.0,'new_obs':empty_obs,'d':False}]

def Or(f):
	if(len(f) == 0): return False
	return f[0] or Or(f[1:])

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
    q_sa.append(reg['q_sa'])
    ob.append(reg['obs'])
    act.append(reg['act'])
    r.append(reg['r'])
    new_ob.append(reg['new_obs'])
    d.append(reg['d'])
  return q_sa, ob, act, r, new_ob, d

def set_rep_mem(env):
	global mem
	obs = np.expand_dims(env.reset(),axis=0)
	obs = ob_frames*obs.tolist()
	for i in range(replay_len):
		a = env.action_space.sample()
		new_obs,r,d = step(env,a)
		mem[i] = {'q_sa': 0.0,'obs':obs,'act':a,'r':r,'new_obs':new_obs,'d':d}
		if d:
			obs = np.expand_dims(env.reset(),axis=0)
			obs = ob_frames*obs.tolist()
		else:
		  obs = new_obs

def step(env,a,render=False):
	obs = np.zeros((ob_frames,state_dim))
	r = np.zeros(ob_frames)
	d = ob_frames*[False]
	for i in range(ob_frames):
		if render: env.render()
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
fc1 = tf.layers.dense(fc1,100,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,20,activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2,1)

loss = tf.losses.mean_squared_error(fc3,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

env = gym.make(env_name)
gamma = 0.99
e = 0.1

with tf.Session() as sess:
	saver = tf.train.Saver()
	if os.path.isfile(model_fn+'.meta'):
		saver.restore(sess,model_fn)
	else:
		sess.run(tf.global_variables_initializer())
	set_rep_mem(env)
	scores = deque(maxlen=100)
	scores_ = []
	for t in range(5000):
		obs = np.expand_dims(env.reset(),axis=0)
		obs = ob_frames*obs.tolist()
		d = False
		s_r = 0.0
		while not d:
			Q = sess.run(fc3,feed_dict={X:num_keys*[obs],act:all_actions})
			Q = np.transpose(Q)[0]
			a = np.argmax(Q)
			print(a)
			new_obs,r,d = step(env,a,t%100==99)
			new_mem = {'q_sa': Q[a],'obs':obs,'act':a,'r':r,'new_obs':new_obs,'d':d}
			s_r += r
			replace_mem(new_mem)
			obs = new_obs
		scores.append(s_r)
		scores_.append(s_r)
		# Replay
		q_sa, b_ob, b_act, b_r, b_new_ob, b_d = get_batch()
		rep_obs = np.repeat(b_new_ob,num_keys,axis=0)	#[np.tile(ob,(num_keys,1)) for ob in b_new_ob]
		Q = sess.run(fc3,feed_dict={X:rep_obs,act:rep_all_actions})
		y = np.zeros(batch_size)
		for j in range(batch_size):
			if b_d[j]:
				y[j] = b_r[j]
			else:
				y[j] = b_r[j]+gamma*np.max(Q[j*(num_keys):(j+1)*num_keys])
		y = np.transpose(y)
		sess.run(train,feed_dict={X:b_ob,act:[all_actions[a] for a in b_act],Y:y})
		if t%500 == 499:
			saver.save(sess,model_fn)

