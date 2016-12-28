from .reinforce import PolicyGradientREINFORCE


import tensorflow as tf
import numpy as np
import pickle
from collections import deque

batch_size = 50
learning_rate = 1e-2
gamma = 0.99
running_reward = None

strategy = 1

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)
writer = tf.train.SummaryWriter("prescribe")
env_name = "mimic"


def jaccard(s1, s2):
    intersection = float(len(s1.intersection(s2)))
    union = len(s1.union(s2))
    if union == 0:
        return 0
    else:
        return intersection / union


class PolicyGradient():
    def __init__(self):
        self.session = tf.InteractiveSession()
        self.train_set = []
        self.dev_set = []
        self.input_vocab = None
        self.output_vocab = None
        self.input_vocab_size = 0
        self.output_vocab_size = 0

        self.PAD_ID = 4127
        self.GO_ID = 4128
        self.EOS_ID = 4129
        self.UNK_ID = 4130

    def load_data(self, input_vocab, output_vocab, train_set, test_set):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_vocab_size = len(input_vocab)
        self.output_vocab_size = len(output_vocab) + 4
        self.train_set = train_set
        self.dev_set = test_set

        self.PAD_ID = self.output_vocab_size - 1
        self.GO_ID = self.PAD_ID - 1
        self.EOS_ID = self.PAD_ID - 2
        self.UNK_ID = self.PAD_ID - 3

    def train(self):
        env = Environment(self.session, self.input_vocab, self.output_vocab)
        episode_history = deque(maxlen=1000)

        while True:
            for idx in range(len(self.train_set)):
                state = env.load_episode(self.train_set[idx])
                total_rewards = 0
                t = 0
                num_pos = 0
                print(idx, env.input, env.output)

                while True:
                    if t > env.max_step: # and num_pos > 0:
                        break
                        # for t in range(env.max_step):
                    action = env.solver.sampleAction(state.transpose(), env.mask.transpose())
                    next_state, reward, done, flag = env.step(action)
                    if flag:
                        num_pos += 1
                        print(t, "pos", action, num_pos, total_rewards)
                    if t % 100 == 0:
                        print(t, action, num_pos, total_rewards)

                    total_rewards += reward
                    env.solver.storeRollout(state, action, reward)

                    state = next_state

                    t += 1

                    env.solver.updateModel()

                    if done:
                        print(total_rewards)
                        break

                env.solver.updateModel()
                env.solver.cleanUp()

                episode_history.append(total_rewards)
                mean_rewards = np.mean(episode_history)

                print("Episode {}".format(idx))
                print("Finished after {} timesteps".format(t+1))
                print("Reward for this episode: {}".format(total_rewards))
                print("Average reward for last 100 episodes: {}".format(mean_rewards))
                if mean_rewards >= 195.0 and len(episode_history) >= 100:
                    print("Environment {} solved after {} episodes".format(env_name, idx+1))
                    break


class Environment(object):
    def __init__(self, sess, vocab_input, vocab_output):
        self.tf_session = sess
        self.vocab_input = vocab_input
        self.vocab_output = vocab_output

        self.idx_to_input = {}
        self.idx_to_output = {}

        for code in self.vocab_input:
            self.idx_to_input[self.vocab_input[code]] = code
        for code in self.vocab_output:
            self.idx_to_output[self.vocab_output[code]] = code

        self.state_dim = len(vocab_input) + len(vocab_output)
        self.num_actions = len(vocab_output) + 1
        self.hidden_dim = 100
        self.max_step = 500

        self.episode = None
        self.state = None
        self.input = set()
        self.output = set()
        self.actions = []
        self.action_offset = len(vocab_input)
        self.terminal_action = self.num_actions - 1

        state_dim, hidden_dim, num_actions = self.state_dim, self.hidden_dim, self.num_actions

        def policy_network(states):
            with tf.device('/gpu:0'):
                print("constructing policy network")
                W1 = tf.get_variable("W1", [state_dim, hidden_dim], initializer=tf.random_normal_initializer())
                b1 = tf.get_variable("b1", [hidden_dim], initializer=tf.constant_initializer(0))
                h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
                W2 = tf.get_variable("W2", [hidden_dim, num_actions], initializer=tf.random_normal_initializer(stddev=0.1))
                b2 = tf.get_variable("b2", [num_actions], initializer=tf.constant_initializer(0))
                p = tf.matmul(h1, W2) + b2
                return p

        self.solver = PolicyGradientREINFORCE(sess,
                                              optimizer,
                                              policy_network,
                                              state_dim,
                                              num_actions,
                                              summary_writer=writer)

    def load_episode(self, episode):
        self.episode = episode
        if "0" in episode[1]:
            episode[1].remove("0")
        self.input = set([self.vocab_input[code] for code in episode[0]])
        self.output = set([self.vocab_output[code] for code in episode[1]])
        self.state = np.zeros((self.state_dim, 1))
        self.actions = []
        self.mask = np.ones((self.num_actions, 1))
        for idx in self.input:
            self.state[idx] = 1
        return self.state

    def step(self, action):
        reward = 0
        done = False
        flag = False

        actions = set(self.actions)
        self.mask[action] = 0

        if action == self.terminal_action:
            done = True
            if strategy == 0:
                reward = jaccard(self.output, actions)
            elif strategy == 1:
                reward = 0
        else:
            if action in self.output and action not in actions:
                flag = True
                if strategy == 0:
                    reward = 0
                elif strategy == 1:
                    reward = 10
            else:
                if strategy == 0:
                    reward = 0
                elif strategy == 1:
                    reward = -1
            self.state[self.action_offset + action] = 1
        self.actions.append(action)
        return self.state, reward, done, flag


def load_data():
    episodes = pickle.load(open("mimic_episodes_train.pkl", "rb"))
    input_vocab = pickle.load(open("diag_vocab.pkl", "rb"))
    output_vocab = pickle.load(open("drug_vocab.pkl", "rb"))
    return episodes, input_vocab, output_vocab


def discount_rewards(r):
    discount_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discount_r[t] = running_add
    return discount_r


def run():
    sess = tf.Session()

    episodes, input_vocab, output_vocab = load_data()
    env = Environment(sess, input_vocab, output_vocab)
    episode_history = deque(maxlen=1000)

    while True:
        for idx in range(1):
            state = env.load_episode(episodes[idx])
            total_rewards = 0
            t = 0
            num_pos = 0
            print(idx, env.input, env.output)

            while True:
                if t > env.max_step: # and num_pos > 0:
                    break
                    # for t in range(env.max_step):
                action = env.solver.sampleAction(state.transpose(), env.mask.transpose())
                next_state, reward, done, flag = env.step(action)
                if flag:
                    num_pos += 1
                    print(t, "pos", action, num_pos, total_rewards)
                if t % 100 == 0:
                    print(t, action, num_pos, total_rewards)

                total_rewards += reward
                env.solver.storeRollout(state, action, reward)

                state = next_state

                t += 1

                env.solver.updateModel()

                if done:
                    print(total_rewards)
                    break

            env.solver.updateModel()
            env.solver.cleanUp()

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)

            print("Episode {}".format(idx))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {}".format(mean_rewards))
            if mean_rewards >= 195.0 and len(episode_history) >= 100:
                print("Environment {} solved after {} episodes".format(env_name, idx+1))
                break

