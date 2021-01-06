import tensorflow as tf
from tensorflow.keras import layers
import gym
import numpy as np
from gym.envs.classic_control import rendering

# rgb = env.render('rgb_array')
        # upscaled=repeat_upsample(rgb)
        # viewer.imshow(upscaled)

def repeat_upsample(rgb_array, k=6, l=6, err=[]):
    if k <= 0 or l <= 0: 
        return rgb_array
    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(600, activation='tanh'))
    model.add(layers.Dense(300))
    model.add(layers.Dense(70))
    model.add(layers.Dense(9,activation='sigmoid'))
    return model

loss_function = tf.keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam()

viewer = rendering.SimpleImageViewer()
env = gym.make('MsPacman-v0')
state = env.reset()

done = False

reward_history = []
state_history = []
action_history = []
max_moves = 40
random_move_prob = 0.9
model_target = create_model()

while not done:

    if np.random.uniform() < random_move_prob:
        action = env.action_space.sample()
    else:
        state_tensor = np.array(state)
        state_tensor = np.ravel(state_tensor)
        state_tensor = np.expand_dims(state_tensor, 0)
        print("\nstate: "+state_tensor.__str__()+"\n")
        action_probs = model_target(state_tensor)
        action = tf.argmax(action_probs, axis=1)
    
    print(action)

    action_history.append(action)
    state_history.append(state)

    observation, reward, done, info = env.step(action)

    reward_history.append(reward)

    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb)
    viewer.imshow(upscaled)

    if len(action_history) > max_moves:
        rewarded_actions = []
        for i in range(len(action_history)):
            if tf.reduce_sum(action_history[i]) > 5:
                rewarded_actions.append(i)
        for i in rewarded_actions:
            goal_action = tf.one_hot(action_history[i],9)
            state_tensor = np.array(state_history[i])
            state_tensor = np.ravel(state_tensor)
            state_tensor = np.expand_dims(state_tensor, 0)
            with tf.GradientTape() as tape:
                action_target_probs = model_target(state_tensor)
                loss = loss_function(goal_action, action_target_probs)
            gradients = tape.gradient(loss , model_target.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_target.trainable_weights))
        action_history = []
        state_history = []
        reward_history = []
        random_move_prob -= 0.05
        
            
            
        
    


