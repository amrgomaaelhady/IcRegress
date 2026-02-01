
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import os
import shutil
from env import ArmEnv
import random
import pandas as pd
import numpy as np
import os
import datetime
from functools import reduce
import math
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import re
warnings.filterwarnings('ignore')
np.random.seed(1)
tf.random.set_random_seed(1)



import os
import glob

# Fetch the only available CSV file in the current directory
current_folder = os.getcwd()
csv_files = glob.glob(os.path.join(current_folder, "*.csv"))

# Check if there's only one CSV file
if len(csv_files) != 1:
    raise ValueError("There should be exactly one CSV file in the current directory.")

# Use the only CSV file found
FILE = csv_files[0]
FILE = os.path.basename(FILE)
#FILE = "Group1_P28_638_1254_data.csv"
numbers = re.findall(r'\d+', FILE)

# Convert to integers and assign to TRAIN_IND and TEST_IND
TRAIN_IND = int(numbers[2]) - 1  # 638 - 1
TEST_IND = int(numbers[3]) - 1   # 1254 - 1

MAX_EPISODES = TRAIN_IND
MAX_EP_STEPS = 50
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 100
REPLACE_ITER_C = 200
MEMORY_CAPACITY = 1000
BATCH_SIZE = 64
print(TRAIN_IND, TEST_IND)

VAR_MIN = 0.1
RENDER = False
LOAD = False



ww = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547]
MODE = ['easy', 'hard']
n_model = 1

env = ArmEnv(mode=MODE[n_model])
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)
           # print('self.a', self.a)
            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
               # print('actor.actions', actions)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
               # print('actor.scaled_a', scaled_a)
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]  # tensor of gradients of each sample (None, a_dim)
        self.replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l4',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.replace)
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)
#M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
saver = tf.train.Saver()
path = './' + MODE[n_model]
'''
if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())'''


def train():
    var = 2.  # control exploration
    import random
    start_range = 1
    end_range = TRAIN_IND

    # Calculate the total count of numbers in the range
    total_numbers = end_range - start_range + 1

    # Generate a list of random numbers covering the entire range without repetition
    random_numbers = random.sample(range(start_range, end_range + 1), total_numbers)
    for ep1 in range(MAX_EPISODES):

        '''if(ep1<  429):
            ep = ep1
        else:'''
        ep = random_numbers[ep1]

        s = env.reset(ep)
       # print(s)
        ep_reward = 0

        for t in range(MAX_EP_STEPS):
            # while True:
            if RENDER:
                env.render(ep)

            # Added exploration noise
            #print('s', s)
            a = actor.choose_action(s)
            #print('actor.a', a)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)  # add randomness to action selection for exploration
            #print('np.clip. a', a)
            s_, r, done = env.step(a, ep)
            #print('s_', s_)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var * .9999, VAR_MIN])  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS - 1 or done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep1,' ',ep,' ',
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./' + MODE[n_model], 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval(q,w):
#    env.set_fps(30)
    #ind = random.randint(0, 320)
    s = env.reset(0)
    ind = TRAIN_IND+1
    p_values = []
    df1 = pd.read_csv(FILE)
    #print(s)
    while True:
            s = env.reset(ind)
            if RENDER:
               env.render(ind)
            a = actor.choose_action(s)
            prediction = a[0]
            p_values.append(a[0])
            s_, r, done = env.step(a,ind)
            #print(prediction)
            #s = s_
            ind =ind + 1
            if(ind> TEST_IND):
                target = df1["gtv_average"][(TRAIN_IND+1):]
                tt = mean_squared_error(target, p_values)
                hbw = abs(df1['gtv_min_average'][TRAIN_IND+1:] - df1['gtv_max_average'][TRAIN_IND+1:]) / 2
                tt1 = abs(p_values - target) / hbw
                tt2 = sum(tt1) / len(p_values)
                count = 0
                oc1 = 0
                oc2 = 0
                oc3 = 0
                oc4 = 0
                oc5 = 0
                ac1 = 0
                ac2 = 0
                ac3 = 0
                ac4 = 0
                ac5 = 0
                mrde1,mrde2,mrde3,mrde4 = 0,0,0,0
                rmse1,rmse2,rmse3,rmse4 = 0,0,0,0
                selected_i_indices = []
                for i in range((TRAIN_IND+1), TEST_IND):

                    if (df1["regularity_new"][i]) == 1:
                        oc1 = oc1 + 1
                        mrde1 = mrde1 + tt1[i]
                        rmse1= rmse1 + ((p_values[i - (TRAIN_IND + 1)] - target[i]) * (p_values[i - (TRAIN_IND + 1)] - target[i]))
                       # rmse1= rmse1 + np.sqrt(tt[i])
                    if (df1["regularity_new"][i]) == 2:
                        oc2 = oc2 + 1
                        mrde2 = mrde2 + tt1[i]
                        rmse2= rmse2 + (p_values[i - (TRAIN_IND + 1)] - target[i])* (p_values[i - (TRAIN_IND + 1)] - target[i])
                       # rmse2 = rmse2 + np.sqrt(tt[i])
                    if (df1["regularity_new"][i]) == 3:
                        oc3 = oc3 + 1
                        mrde3 = mrde3 + tt1[i]
                        rmse3= rmse3 + (p_values[i - (TRAIN_IND + 1)] - target[i])* (p_values[i - (TRAIN_IND + 1)] - target[i])
                       # rmse3 = rmse3 + np.sqrt(tt[i])
                    if (df1["regularity_new"][i]) == 6:
                        oc4 = oc4 + 1
                        mrde4 = mrde4 + tt1[i]
                        #print(i-TRAIN_IND+1, len(p_values))
                        rmse4 = rmse4 + (p_values[i - (TRAIN_IND + 1)] - target[i])* (p_values[i - (TRAIN_IND + 1)] - target[i])

                       # rmse4 = rmse4 + np.sqrt(tt[i])
                    if tt1[i] > 1.0:
                        if (df1["regularity_new"][i]) == 1:
                            ac1 = ac1 + 1
                        if (df1["regularity_new"][i]) == 2:
                            ac2 = ac2 + 1
                        if (df1["regularity_new"][i]) == 3:
                            ac3 = ac3 + 1
                        if (df1["regularity_new"][i]) == 6:
                            ac4 = ac4 + 1
                        selected_i_indices.append(i)
                        count = count + 1
                        # print(df1["difference"][i])
                rmse1 = rmse1 * (180 / math.pi)
                rmse2 = rmse2 * (180 / math.pi)
                rmse3 = rmse3 * (180 / math.pi)
                rmse4 = rmse4 * (180 / math.pi)
                tt = tt * (180 / math.pi)

                #print(oc1, oc2, oc3, oc4)
                print('accuracy', 100- (ac1 * 100 / oc1) , 100 - (ac2 * 100 / oc2) , 100- (ac3 * 100 / oc3) , 100- (ac4 * 100 / oc4))
                print('mrde', mrde1/ oc1, mrde2/oc2, mrde3/oc3, mrde4/oc4)
                print('rmse', np.sqrt(rmse1/oc1), np.sqrt(rmse2/oc2),np.sqrt(rmse3/oc3), np.sqrt(rmse4/oc4))
                print('avg acc rmse mrde', 100- ((ac1 * 100 / oc1 )+ ( ac2 * 100 / oc2) +  (ac3 * 100 / oc3) + (ac4 * 100 / oc4 ) ) / 4, np.sqrt(tt), tt2)
                print(q, w )
                with open('results.txt', 'a') as file:
                    # Write text to the file
                    file.write(f"{q} {w} {np.sqrt(tt)} {tt2}\n")
                #df8['asd'] = p_values[:306]
                #df8.to_csv("Pre_.csv", index=False)
                break

def eval1(q,w):
#    env.set_fps(30)
    #ind = random.randint(0, 320)
    s = env.reset(0)
    #df2 = pd.read_csv("Refined_values.csv")
    ind = 0
    p_values = []
    df1 = pd.read_csv("Grop1_50_p17_data.csv")
    df1 = df1[0: 1647]
    #print(s)
    while True:
            s = env.reset(ind)
            if RENDER:
               env.render(ind)
            a = actor.choose_action(s)
            prediction = a[0] * 50
            p_values.append(prediction)
            s_, r, done = env.step(a,ind)
            #print(prediction)
            #s = s_
            ind =ind + 1
            if(ind> 1646):
                df2 = df1
                target = df1["target"]
                #print(p_values[0], target[1622], df2["target"][0])
                tt1 = abs(p_values - target) / df1['h_B_w']
                tt1_reset = tt1.reset_index(drop=True)
                df2["mre"] = tt1_reset
                #print(tt1_reset[0])
                #print(tt1[1622])
                #print(df2["mre"][0])
                #df2["mre"] = ms
                df2["predict"] = p_values
                df2.to_csv('predicted_data_train.csv', index=False)
                #print(sum(tt))
                #print(len(p_values))

                selected_i_indices = []
                ##for i in range(1622, 2165):

                # Save the new DataFrame to a CSV file
                #new_df.to_csv('selected_data.csv', index=False)
                break

def eval_p(q,w):
#    env.set_fps(30)
    #ind = random.randint(0, 320)
    s = env.reset(0)
    ind = TRAIN_IND+1
    p_values = []
    df1 = pd.read_csv(FILE)
    #print(s)
    while True:
            s = env.reset(ind)
            if RENDER:
               env.render(ind)
            a = actor.choose_action(s)
            prediction = a[0]
            p_values.append(a[0])
            s_, r, done = env.step(a,ind)
            #print(prediction)
            #s = s_
            ind =ind + 1
            if(ind> TEST_IND):
                target = df1["gtv_average"][(TRAIN_IND+1):]
                tt = mean_squared_error(target, p_values)
                hbw = abs(df1['gtv_min_average'][TRAIN_IND+1:] - df1['gtv_max_average'][TRAIN_IND+1:]) / 2
                tt1 = abs(p_values - target) / hbw
                tt2 = sum(tt1) / len(p_values)
                participants = ['P58', 'P60']

                for p in participants:
                    count = 0
                    oc1 = 0
                    oc2 = 0
                    oc3 = 0
                    oc4 = 0
                    oc5 = 0
                    ac1 = 0
                    ac2 = 0
                    ac3 = 0
                    ac4 = 0
                    ac5 = 0
                    mrde1,mrde2,mrde3,mrde4 = 0,0,0,0
                    rmse1,rmse2,rmse3,rmse4 = 0,0,0,0
                    selected_i_indices = []
                    for i in range((TRAIN_IND+1), TEST_IND):

                        if (df1["participant"][i]) == p:
                            oc1 = oc1 + 1
                            mrde1 = mrde1 + tt1[i]
                            rmse1= rmse1 + ((p_values[i - (TRAIN_IND + 1)] - target[i]) * (p_values[i - (TRAIN_IND + 1)] - target[i]))
                           # rmse1= rmse1 + np.sqrt(tt[i])
                        if (df1["regularity_new"][i]) == 2:
                            oc2 = oc2 + 1
                            mrde2 = mrde2 + tt1[i]
                            rmse2= rmse2 + (p_values[i - (TRAIN_IND + 1)] - target[i])* (p_values[i - (TRAIN_IND + 1)] - target[i])
                           # rmse2 = rmse2 + np.sqrt(tt[i])
                        if (df1["regularity_new"][i]) == 3:
                            oc3 = oc3 + 1
                            mrde3 = mrde3 + tt1[i]
                            rmse3= rmse3 + (p_values[i - (TRAIN_IND + 1)] - target[i])* (p_values[i - (TRAIN_IND + 1)] - target[i])
                           # rmse3 = rmse3 + np.sqrt(tt[i])
                        if (df1["regularity_new"][i]) == 6:
                            oc4 = oc4 + 1
                            mrde4 = mrde4 + tt1[i]
                            #print(i-TRAIN_IND+1, len(p_values))
                            rmse4 = rmse4 + (p_values[i - (TRAIN_IND + 1)] - target[i])* (p_values[i - (TRAIN_IND + 1)] - target[i])

                           # rmse4 = rmse4 + np.sqrt(tt[i])
                        if tt1[i] > 1.0:
                            if (df1["participant"][i]) == p:
                                ac1 = ac1 + 1
                            if (df1["regularity_new"][i]) == 2:
                                ac2 = ac2 + 1
                            if (df1["regularity_new"][i]) == 3:
                                ac3 = ac3 + 1
                            if (df1["regularity_new"][i]) == 6:
                                ac4 = ac4 + 1
                            selected_i_indices.append(i)
                            count = count + 1
                            # print(df1["difference"][i])
                    rmse1 = rmse1 * (180 / math.pi)
                    rmse2 = rmse2 * (180 / math.pi)
                    rmse3 = rmse3 * (180 / math.pi)
                    rmse4 = rmse4 * (180 / math.pi)
                    tt = tt * (180 / math.pi)
                    print(f"{p} {100 - (ac1 * 100 / oc1):.2f}")
                    #print(oc1, oc2, oc3, oc4)
                    #print('accuracy', 100- (ac1 * 100 / oc1) , 100 - (ac2 * 100 / oc2) , 100- (ac3 * 100 / oc3) , 100- (ac4 * 100 / oc4))
                    #print('mrde', mrde1/ oc1, mrde2/oc2, mrde3/oc3, mrde4/oc4)
                    #print('rmse', np.sqrt(rmse1/oc1), np.sqrt(rmse2/oc2),np.sqrt(rmse3/oc3), np.sqrt(rmse4/oc4))
                    #print('avg acc rmse mrde', 100- ((ac1 * 100 / oc1 )+ ( ac2 * 100 / oc2) +  (ac3 * 100 / oc3) + (ac4 * 100 / oc4 ) ) / 4, np.sqrt(tt), tt2)
                    #print(q, w )
                with open('results.txt', 'a') as file:
                    # Write text to the file
                    file.write(f"{q} {w} {np.sqrt(tt)} {tt2}\n")
                #df8['asd'] = p_values[:306]
                #df8.to_csv("Pre_.csv", index=False)
                break
if __name__ == '__main__':

    #train()

    for i in range (22, 150, 2):
        for j in range(5500, 8500, 250):
            MAX_EP_STEPS = int(i)
            MEMORY_CAPACITY = int(j)
            M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)
            print(i, j)
            LOAD = False
            f= "./base_model"
            #saver.restore(sess, tf.train.latest_checkpoint(f))
            #sess.run(tf.global_variables_initializer())
           # BATCH_SIZE = 128
            #train()
            LOAD = True
            saver.restore(sess, tf.train.latest_checkpoint(path))
            eval(i,j)
            eval_p(i,j)
            break
        break
