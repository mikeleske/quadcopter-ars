# AI 2018

# Importing the libraries
import numpy as np
from task import Task

# Setting the Hyper Parameters

class Hp():

    def __init__(self):
        self.nb_steps = 200
        self.episode_length = 1000
        self.learning_rate = 0.01 #0.02
        self.nb_directions = 16
        self.nb_best_directions = 4
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.3
        self.seed = 1

# Normalizing the states

class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Building the AI

class Policy():

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)

    def sample_deltas(self):
        return [np.random.rand(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Building a tracer template

class Tracer():
    
    def __init__(self):
        self.rewards = []
        self.states = []
        self.actions = []
        self.traces = []
    
    def add_trace(self, trace):
        self.traces.append(trace)
        
# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    tt = Tracer()
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = abs(policy.evaluate(state, delta, direction) * env.action_high).clip(env.action_low, env.action_high)
        state, reward, done = env.takeoff(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
        
        # Tracing
        if not direction:
            tt.actions.append(action.tolist())
            tt.rewards.append(reward)
            tt.states.append(state.tolist())
    if not direction:
        t.traces.append(tt)
    return sum_rewards

# Training the AI

def train(env, policy, normalizer, hp):
    global best_reward

    for step in range(hp.nb_steps):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()

        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])

        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        if reward_evaluation > best_reward:
            best_reward = reward_evaluation
            print('Step {:3}/{} with best reward: {:8.3f}. Saving policy theta.'.format(step, hp.nb_steps, best_reward))
            np.save('theta', policy.theta)
        #print('Step:', step, 'Reward:', reward_evaluation, 'Final position:', env.sim.pose[:3])

def run(env, policy, normalizer, hp):
    reward_evaluation = explore(env, normalizer, policy)

# Running the main code

import warnings
warnings.filterwarnings('ignore')

hp = Hp()
t = Tracer()
#np.random.seed(hp.seed)

target_pos = np.array([0., 0., 150.])
env = Task(target_pos=target_pos)
nb_inputs = env.state_size
nb_outputs = env.action_size
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)

best_reward = -999.
train(env, policy, normalizer, hp)
#policy.theta = np.load('theta.npy')
run(env, policy, normalizer, hp)
