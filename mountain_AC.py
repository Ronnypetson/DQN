import tensorflow as tf
import gym
import numpy as np
import os
import random

env_name = 'MountainCar-v0'
state_dim = 2
ob_frames = 1
num_keys = 3
learning_rate = 0.01
batch_size = 32
replay_len = 1000
oldest_mem = 0
default_action = 1
empty_obs = np.zeros((ob_frames,state_dim))
mem = replay_len*[{'q_sa':0.0,'obs':empty_obs,'act':default_action,'r':0.0,'new_obs':empty_obs,'d':False}]
model_fn = 'checkpoint/'+env_name+'/'+env_name+'.ckpt'


