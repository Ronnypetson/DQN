import tensorflow as tf
import gym
import numpy as np
import os
import random
from collections import deque
from matplotlib import pyplot as plt

env_name = 'MountainCarContinuous-v0'
state_dim = 2
ob_frames = 3
num_keys = 1
learning_rate = 0.001
batch_size = 32
replay_len = 10000
oldest_mem = 0
default_action = 1
empty_obs = np.zeros((ob_frames,state_dim))
mem = replay_len*[{'q_sa':0.0,'obs':empty_obs,'act':default_action,'r':0.0,'new_obs':empty_obs,'d':False}]
model_fn = 'checkpoint/'+env_name+'/'+env_name+'.ckpt'

def Or(f):
	if(len(f) == 0): return False
	return f[0] or Or(f[1:])

def rargmax(a):
	max_a = np.max(a)
	maxes = np.argwhere(a == max_a)
	return random.choice([a_[0] for a_ in maxes])

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

def step(env,a,render=False):
	obs = np.zeros((ob_frames,state_dim))
	r = np.zeros(ob_frames)
	d = ob_frames*[False]
	for i in range(ob_frames):
		if render: env.render()
		obs[i],r[i],d[i],_ = env.step(a)
		if d[i]: break
	r = np.sum(r)
	d = Or(d)
	return obs,r,d

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

# Actor
X = tf.placeholder(tf.float32,[None,ob_frames,state_dim])
X_ = tf.contrib.layers.flatten(X)
a1 = tf.layers.dense(X_,50,activation=tf.nn.relu)
a2 = tf.layers.dense(a1,15,activation=tf.nn.relu)
A = tf.layers.dense(a2,num_keys) # activation=None

# Critic
A_ = tf.placeholder(tf.float32,[None,num_keys])
XA = tf.concat([X_,A_],1)
fc1 = tf.layers.dense(XA,50,activation=tf.nn.relu)
fc2 = tf.layers.dense(fc1,10,activation=tf.nn.relu)
fc3 = tf.layers.dense(fc2,1)
Y = tf.placeholder(tf.float32,None)

loss = tf.losses.mean_squared_error(Y,fc3)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

actorLoss = -tf.reduce_mean(Y)*(tf.norm(A)/tf.norm(A))
actorTrain = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(actorLoss)

# Auxiliary network
#a1_ = tf.layers.dense(X_,50,activation=tf.nn.relu)
#a2_ = tf.layers.dense(a1_,15,activation=tf.nn.relu)
#A_ = tf.layers.dense(a2_,num_keys)
#fc1_ = tf.layers.dense(XA,50,activation=tf.nn.relu)
#fc2_ = tf.layers.dense(fc1_,10,activation=tf.nn.relu)
#fc3_ = tf.layers.dense(fc2_,1)

# Copy ops
#vars_cp = tf.trainable_variables()
#copy_ops = [vars_cp[ix+len(vars_cp)//2].assign(var.value()) for ix, var in enumerate(vars_cp[0:len(vars_cp)//2])]

env = gym.make(env_name)
gamma = 0.99

with tf.Session() as sess:
	saver = tf.train.Saver()
	if os.path.isfile(model_fn+'.meta'):
		saver.restore(sess,model_fn)
	else:
		sess.run(tf.global_variables_initializer())
		#map(lambda x: sess.run(x), copy_ops)
	set_rep_mem(env)
	scores = deque(maxlen=100)
	scores_ = []
	for t in range(5000):
		obs = np.expand_dims(env.reset(),axis=0)
		obs = ob_frames*obs.tolist()
		d = False
		s_r = 0.0
		while not d:
			actA = sess.run(A,feed_dict={X:[obs]})
			actQ = sess.run(fc3,feed_dict={X:[obs],A_:actA})
			actQ = np.transpose(actQ)
			actA = np.transpose(actA)
			a = actA[0] + random.gauss(0.0,0.1)
			print(a)
			new_obs,r,d = step(env,a,t%100==99)
			new_mem = {'q_sa': actQ[0],'obs':obs,'act':a,'r':r,'new_obs':new_obs,'d':d}
			s_r += r
			replace_mem(new_mem)
			obs = new_obs
			# Replay
			q_sa, b_ob, b_act, b_r, b_new_ob, b_d = get_batch()
			actA = sess.run(A,feed_dict={X:b_new_ob})
			actQ = sess.run(fc3,feed_dict={X:b_new_ob,A_:actA})
			y = np.zeros(batch_size)
			for j in range(batch_size):
				if b_d[j]:
					y[j] = b_r[j]
				else:
					y[j] = b_r[j]+gamma*actQ[j]
			actA = sess.run(A,feed_dict={X:b_ob})
			sess.run(train,feed_dict={X:b_ob,A_:actA,Y:y})
			sess.run(actorTrain,feed_dict={X:b_ob,Y:actQ})
			#map(lambda x: sess.run(x), copy_ops)
		scores.append(s_r)
		scores_.append(s_r)
		if t%1000 == 999:
			saver.save(sess,model_fn)

