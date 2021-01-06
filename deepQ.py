import tensorflow as tf
import gym
from gym.envs.classic_control import rendering
import numpy as np
import sys
from tensorflow.keras import layers

def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    if k <= 0 or l <= 0: 
        return rgb_array
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)
# setup
viewer = rendering.SimpleImageViewer()
env = gym.make('MsPacman-v0')
env.reset()
np.set_printoptions(threshold=sys.maxsize)

# neural network
class network(tf.keras.Model):

    def __init__(self):
        super.__init__(self)
        self.flatten

upscale = 4
frames = 200

for i in range(frames):
    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb, upscale, upscale)
    viewer.imshow(upscaled)
    observation, reward, done, info = env.step(9)
