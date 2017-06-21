#!/usr/bin/env python

import math
import random
import atexit
import os

import time
import datetime
import numpy as np
import rospy
import gym
from gym import wrappers

import tensorflow as tf
from rotors_reinforce.srv import GetState
from rotors_reinforce.srv import PerformAction

# STATE_SIZE = 3 + 4 + 3 + 3
STATE_SIZE = 4 * (1)
ACTION_NUM = 1
ACTION_SIZE = 2

SCALE = 100

env = gym.make('CartPole-v0')
env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

def termination_funk():
    print "terminated\n"
    _train_writer.close()

    # save_path = saver.save(sess, "./tmp/model.ckpt")
    # print("Model saved in file: %s" % save_path)

atexit.register(termination_funk)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        tf.summary.scalar('sum', tf.reduce_sum(var))


class ComputationalGraph:
    def __init__(self):
        self.learning_rate = 0.1
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * SCALE

        self.hid_neurons_num = 16
        self.w = {}
        self.t_w = {}

    def constructGraph(self, sess, state_size, action_num, action_size):
        with tf.variable_scope("Q-learning_main"):
            self.ql_state = tf.placeholder(tf.float32, [None, state_size], name="ql_state")

            # formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
            #hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2

            #feature extraction

            self.w['W1'] = tf.get_variable("ql_W1_main", [state_size, action_num * action_size], initializer=tf.random_normal_initializer(stddev=0.02))
            self.w['b1'] = tf.get_variable("ql_b1_main", [action_num * action_size], initializer=tf.constant_initializer(0.0))

            self.w['W2'] = tf.get_variable("ql_W2_main", [self.hid_neurons_num, action_num * action_size], initializer=tf.random_normal_initializer(stddev=0.02))
            self.w['b2'] = tf.get_variable("ql_b2_main", [action_num * action_size], initializer=tf.constant_initializer(0.0))

            # self.w['W3'] = tf.get_variable("ql_W3_main", [self.hid_neurons_num, self.hid_neurons_num], initializer=tf.random_normal_initializer(stddev=0.02))
            # self.w['b3'] = tf.get_variable("ql_b3_main", [self.hid_neurons_num], initializer=tf.constant_initializer(0.0))

            self.ql_output1 = (tf.nn.bias_add(tf.matmul(self.ql_state, self.w['W1']), self.w['b1']))
            #self.ql_output2 = (tf.nn.bias_add(tf.matmul(self.ql_output1, self.w['W2']), self.w['b2']))
            # self.ql_output3 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.ql_output2, self.w['W2']), self.w['b2']))

            #self.ql_streamA, self.ql_streamV = tf.split(self.ql_output2, 2, axis=1)

            #duelling

            # self.w['AW1'] = tf.get_variable("ql_AW1_main", [self.hid_neurons_num, self.hid_neurons_num],
            #                              initializer=tf.random_normal_initializer(stddev=0.02))
            # self.w['VW1'] = tf.get_variable("ql_VW1_main", [self.hid_neurons_num, self.hid_neurons_num],
            #                              initializer=tf.random_normal_initializer(stddev=0.02))
            #
            # self.w['Ab1'] = tf.get_variable("ql_Ab1_main", [self.hid_neurons_num], initializer=tf.constant_initializer(0.0))
            # self.w['Vb1'] = tf.get_variable("ql_Vb1_main", [self.hid_neurons_num], initializer=tf.constant_initializer(0.0))
            #
            # self.w['AW2'] = tf.get_variable("ql_AW2_main", [self.hid_neurons_num, action_num * action_size],
            #                              initializer=tf.random_normal_initializer(stddev=0.02))
            # self.w['VW2'] = tf.get_variable("ql_VW2_main", [self.hid_neurons_num, 1],
            #                              initializer=tf.random_normal_initializer(stddev=0.02))
            #
            # self.w['Ab2'] = tf.get_variable("ql_Ab2_main", [action_num * action_size], initializer=tf.constant_initializer(0.0))
            # self.w['Vb2'] = tf.get_variable("ql_Vb2_main", [1], initializer=tf.constant_initializer(0.0))
            #
            # self.ql_adv_out1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.ql_output3, self.w['AW1']), self.w['Ab1']))
            # self.ql_advantage = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.ql_adv_out1, self.w['AW2']), self.w['Ab2']))
            #
            # self.ql_val_out1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.ql_output3, self.w['VW1']), self.w['Vb1']))
            # self.ql_value = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.ql_val_out1, self.w['VW2']), self.w['Vb2']))


            # self.ql_Qout = self.ql_value + tf.subtract(self.ql_advantage,
            #                                            tf.reduce_mean(self.ql_advantage, axis=-1, keep_dims=True))
            self.ql_Qout = self.ql_output1
            self.ql_computated_action = tf.argmax(self.ql_Qout, axis=1)

            self.ql_targetQ = tf.placeholder(tf.float32, [None, action_num * action_size], name="ql_targetQ")
            self.ql_actions = tf.placeholder(tf.int32, [None], name="ql_actions")
            self.actions_one_hot = tf.one_hot(self.ql_actions, ACTION_SIZE, dtype=tf.float32)

            self.ql_Q = tf.reduce_sum(tf.multiply(self.ql_Qout, self.actions_one_hot), axis=-1)

            self.ql_td_error = tf.square(self.ql_targetQ - self.ql_Q)
            self.ql_loss = tf.reduce_mean(self.ql_td_error)

            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.learning_rate_step,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))
            # self.ql_optimizer = tf.train.RMSPropOptimizer(
            #      self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.ql_loss)

            loss = tf.reduce_sum(tf.square(self.ql_targetQ - self.ql_Qout))

            self.ql_optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

            #self.ql_optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_op).minimize(self.ql_loss)

            #self.ql_optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.ql_lrate).minimize(self.ql_loss)

        with tf.variable_scope("Q-learning_target"):
            self.t_state = tf.placeholder(tf.float32, [None, state_size], name="t_state")

            # formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
            # hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2

            # feature extraction

            self.t_w['W1'] = tf.get_variable("t_W1_main", [state_size, self.hid_neurons_num],
                                           initializer=tf.random_normal_initializer(stddev=0.02))
            self.t_w['b1'] = tf.get_variable("t_b1_main", [self.hid_neurons_num],
                                           initializer=tf.constant_initializer(0.0))

            self.t_w['W2'] = tf.get_variable("t_W2_main", [self.hid_neurons_num, action_num * action_size],
                                           initializer=tf.random_normal_initializer(stddev=0.02))
            self.t_w['b2'] = tf.get_variable("t_b2_main", [action_num * action_size],
                                           initializer=tf.constant_initializer(0.0))
            #
            # self.t_w['W3'] = tf.get_variable("t_W3_main", [self.hid_neurons_num, self.hid_neurons_num],
            #                                initializer=tf.random_normal_initializer(stddev=0.02))
            # self.t_w['b3'] = tf.get_variable("t_b3_main", [self.hid_neurons_num],
            #                                initializer=tf.constant_initializer(0.0))

            self.t_output1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_state, self.t_w['W1']), self.t_w['b1']))
            self.t_output2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_output1, self.t_w['W2']), self.t_w['b2']))
            # self.t_output3 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_output2, self.t_w['W2']), self.t_w['b2']))

            # self.t_streamA, self.t_streamV = tf.split(self.t_output2, 2, axis=1)

            # duelling

            # self.t_w['AW1'] = tf.get_variable("t_AW1_main", [self.hid_neurons_num, self.hid_neurons_num],
            #                                 initializer=tf.random_normal_initializer(stddev=0.02))
            # self.t_w['VW1'] = tf.get_variable("t_VW1_main", [self.hid_neurons_num, self.hid_neurons_num],
            #                                 initializer=tf.random_normal_initializer(stddev=0.02))
            #
            # self.t_w['Ab1'] = tf.get_variable("t_Ab1_main", [self.hid_neurons_num],
            #                                 initializer=tf.constant_initializer(0.0))
            # self.t_w['Vb1'] = tf.get_variable("t_Vb1_main", [self.hid_neurons_num],
            #                                 initializer=tf.constant_initializer(0.0))
            #
            # self.t_w['AW2'] = tf.get_variable("t_AW2_main", [self.hid_neurons_num, action_num * action_size],
            #                                 initializer=tf.random_normal_initializer(stddev=0.02))
            # self.t_w['VW2'] = tf.get_variable("t_VW2_main", [self.hid_neurons_num, 1],
            #                                 initializer=tf.random_normal_initializer(stddev=0.02))
            #
            # self.t_w['Ab2'] = tf.get_variable("t_Ab2_main", [action_num * action_size],
            #                                 initializer=tf.constant_initializer(0.0))
            # self.t_w['Vb2'] = tf.get_variable("t_Vb2_main", [1], initializer=tf.constant_initializer(0.0))
            #
            # self.t_adv_out1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_output3, self.t_w['AW1']), self.t_w['Ab1']))
            # self.t_advantage = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_adv_out1, self.t_w['AW2']), self.t_w['Ab2']))
            #
            # self.t_val_out1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_output3, self.t_w['VW1']), self.t_w['Vb1']))
            # self.t_value = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.t_val_out1, self.t_w['VW2']), self.t_w['Vb2']))

            # self.t_Qout = self.t_value + tf.subtract(self.t_advantage,
            #                                            tf.reduce_mean(self.t_advantage, axis=-1, keep_dims=True))
            self.t_Qout = self.t_output2
            self.t_computated_action = tf.argmax(self.t_Qout, axis=1)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    def update_target_q_network(self, sess):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval(session = sess)}, session = sess)
            #sess.run(self.t_w_assign_op[name], feed_dict={})

    def qLearningPrediction(self, sess, state):
        state_2D = np.expand_dims(state, axis=0)
        action, Q = sess.run([self.ql_computated_action, self.ql_Q], feed_dict={self.ql_state: state_2D})
        return action, Q

    def getQ(self, sess, state):
        state_2D = np.expand_dims(state, axis=0)
        Q = sess.run(self.ql_Q, feed_dict={self.ql_state: state_2D})
        return Q

    def optimizeQ(self, sess, state, targetQ):
        state_2D = np.expand_dims(state, axis=0)
        _, W1, W2 = sess.run([self.ql_optimizer, self.ql_W1, self.ql_W2], feed_dict={self.ql_state: state_2D,
                                                                                     self.ql_Qnext: targetQ})

        print W1  # print current weights of network
        print W2

        # def addReward2Summary(self, sess, reward, step):
        #     statistics = sess.run(self.merged, feed_dict={self.episode_rewards: reward})
        #     _train_writer.add_summary(statistics, step)
        #     _train_writer.flush()


class ExperienceBuffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.s1 = []
        self.a = []
        self.s2 = []
        self.r = []
        self.t = []


    def add(self, s1, a, s2, r, t):
        # if len(self.buffer) + len(experience) >= self.buffer_size:
        #     self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        # self.buffer.extend(experience)
        self.s1.append(s1)
        self.a.append(a)
        self.s2.append(s2)
        self.r.append(r)
        self.t.append(t)
        if len(self.t) > self.buffer_size:
            self.s1.pop(0)
            self.a.pop(0)
            self.s2.pop(0)
            self.r.pop(0)
            self.t.pop(0)

    def sample(self, size):
        ret_s1 = []
        ret_s2 = []
        ret_a = []
        ret_r = []
        ret_t = []

        for x in range(0, size):
            i = np.random.randint(len(self.t))
            ret_s1.append(self.s1[i])
            ret_s2.append(self.s2[i])
            ret_a.append(self.a[i])
            ret_r.append(self.r[i])
            ret_t.append(self.t[i])

        return ret_s1, ret_a, ret_s2, ret_r, ret_t


# def updateTargetGraph(tfVars, tau=0.001):
    #TODO: implement safer extraction of the trainable variables
    # total_vars = len(tfVars)
    # op_holder = []
    # for idx, var in enumerate(tfVars[0:total_vars // 2]):
    #     op_holder.append(tfVars[idx + total_vars // 2].assign(
    #         (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    # return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def qlearning_node():
    # set up env
    # rospy.init_node('qLearning', anonymous=True)
    # perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    # get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    graph = ComputationalGraph()

    buffer_size = SCALE * 100
    #buffer_size = 1

    global_buffer = ExperienceBuffer(buffer_size)

    action = np.zeros(ACTION_NUM)
    executed_action = np.zeros(3 + ACTION_NUM)

    global sess
    sess = tf.Session()

    graph.constructGraph(sess, STATE_SIZE, ACTION_NUM, ACTION_SIZE)

    # TODO: implement summary class
    episode_rewards = tf.placeholder(tf.float32, [1], name="episode_rewards")
    variable_summaries(episode_rewards)
    # variable_summaries(mainGraph.ql_W1)
    #variable_summaries(mainGraph.ql_b1)
    merged = tf.summary.merge_all()
    global _train_writer
    _train_writer = tf.summary.FileWriter('./log/train', sess.graph)


    global saver
    saver = tf.train.Saver()
    path = "./tmp/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    # TODO: restore doesn't work. fix it.
    # try:
    #     saver.restore(sess, "./tmp/model.ckpt")
    #     print "model restored"
    # except:
    #     print "model isn't restored. random initialization"
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    trainables = tf.trainable_variables()
    # target_ops = updateTargetGraph(trainables)

    step = 0
    total_step = 0
    maxstep = 100
    batch_size = 32

    startE = 1.0
    endE = 0.1
    e = startE
    #anneling_steps = buffer_size
    anneling_steps = 1000
    stepDrop = (startE - endE) / anneling_steps
    gamma = 0.99

    target_q_update_step = 1 * SCALE
    #pre_train_steps = 5 * SCALE
    pre_train_steps = 100
    update_freq = 4

    episode_number = 0
    total_reward = np.zeros(1)

    current_speed = np.zeros(1)


    # main loop:
    while not rospy.is_shutdown():
        crashed_flag = False

        # episode_buffer = ExperienceBuffer()

        # get initial state
        # rospy.wait_for_service('env_tr_get_state')
        # try:
        #     response = get_state_client()
        # except rospy.ServiceException, e:
        #     print "Service call failed: %s" % e

        state = env.reset()

        #state = np.concatenate([[response.target_position[2] - response.position[2]]])
        state_stack = []
        state_stack.append(state)
        state_stack.append(state)
        state_stack.append(state)
        state_stack.append(state)

        # run episode, while not crashed and simulation is running


        while 1:

            # get most probable variant to act for each action, and the probabilities

            # action = sess.run(mainGraph.ql_computated_action,
            #                   feed_dict={mainGraph.ql_state: [np.array([state_stack]).ravel()]})

            # action = sess.run(graph.ql_computated_action,
            #                  feed_dict={graph.ql_state: [state]})

            #
            # if np.random.rand(1) < e:
            #     action[0] = np.random.randint(ACTION_SIZE)
            # else:
            #     action[0] = int(action[0])

            #executed_action[3] = float(action[0]) / float(ACTION_SIZE - 1)


            #new_state, reward, done, info = env.step(int(action[0]))

            a, allQ = sess.run([graph.ql_computated_action, graph.ql_Qout],feed_dict={graph.ql_state: [state]})
            print a
            print allQ
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            new_state, reward, done, info = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(graph.ql_Qout,feed_dict={graph.ql_state: [new_state]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = reward + gamma * maxQ1
            #Train our network using target and predicted Q values
            w, _ = sess.run([graph.w['W1'], graph.ql_optimizer],feed_dict={graph.ql_state: [state], graph.ql_targetQ :targetQ})

            print w

            time.sleep(0.1)

            # rospy.wait_for_service('env_tr_perform_action')
            # try:
            #     new_response = perform_action_client(executed_action)
            # except rospy.ServiceException, e:
            #     print "Service call failed: %s" % e

            #current_speed[0] = new_response.position[2] - response.position[2]

            #new_state = np.concatenate([[new_response.target_position[2] - new_response.position[2]]])
            state_stack.append(new_state)

            #print new_state

            # episode_buffer.add(
            #    np.reshape(np.array([np.array([state_stack[:-1]]).ravel(), action[0], response.reward, np.array([state_stack[1:]]).ravel(), response.crashed]), [1, 5]))

            global_buffer.add(state, action[0], new_state, reward, done)



            #print np.array([state_stack[:-1]]).ravel()

            state_stack.pop(0)

            total_reward[0] += float(reward)
            state = new_state

            if e > endE:
                e -= stepDrop

            step = step + 1
            total_step = total_step + 1

            if done:
                break

            # print total_step, action[0], " e:", e, "                   \r",

            # if total_step > pre_train_steps:
            #     if e > endE:
            #         e -= stepDrop
            #
            #     if total_step % (update_freq) == 0:
            #         s1, a, s2, r, t = global_buffer.sample(batch_size)
            #
            #         q_t_plus_1 = sess.run(graph.t_Qout, feed_dict={
            #             graph.t_state: s1
            #         })
            #
            #         terminal = np.array(t) + 0.
            #         max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            #
            #         target_q_t = (1. - terminal) * gamma * max_q_t_plus_1 + r
            #
            #         sess.run(graph.ql_optimizer, feed_dict={
            #             graph.ql_state: s1,
            #             graph.ql_targetQ: target_q_t,
            #             graph.ql_actions: a,
            #             graph.learning_rate_step: total_step
            #         })

                    # print "========================"
                    # print tot
                    # print np.vstack(trainBatch[:, 0])
                    # print np.vstack(trainBatch[:, 3])
                    # print target_q_t
                    # print trainBatch[:, 1]

                    # Q1 = sess.run(mainGraph.ql_computated_action,
                    #               feed_dict={mainGraph.ql_state: np.vstack(trainBatch[:, 3])})
                    # Q2 = sess.run(targetGraph.ql_Qout,
                    #               feed_dict={targetGraph.ql_state: np.vstack(trainBatch[:, 3])})
                    #
                    # end_multiplier = -(trainBatch[:, 4] - 1)
                    # doubleQ = Q2[range(batch_size), Q1]
                    # targetQ = trainBatch[:, 2] + (gamma * doubleQ * end_multiplier)
                    #
                    # W1, b1, _ = sess.run([mainGraph.ql_W1, mainGraph.ql_b1, mainGraph.ql_optimizer],
                    #                      feed_dict={
                    #                          mainGraph.ql_state: np.vstack(trainBatch[:, 0]),
                    #                          mainGraph.ql_targetQ: targetQ,
                    #                          mainGraph.ql_actions: trainBatch[:, 1],
                    #                          mainGraph.learning_rate_step: total_step
                    #                      })

                    #print b1
                    #print W1



                #     updateTarget(target_ops, sess)
                #
                # if total_step % target_q_update_step == target_q_update_step - 1:
                #     graph.update_target_q_network(sess)

            # else:
            #     print total_step


                # if total_step % target_q_update_step == 0:
                #     updateTarget(target_ops, sess)
                #     #print W1
                    #print W2




            #response = new_response
            #state = new_state

            #crashed_flag = response.crashed
            #crashed_flag = done



        # targetGraph.addReward2Summary(sess, total_reward, episode_number)
        statistics = sess.run(merged, feed_dict={episode_rewards: total_reward})
        _train_writer.add_summary(statistics, episode_number)
        _train_writer.flush()
        episode_number = episode_number + 1
        total_reward[0] = 0.0

        if episode_number % 500 == 0:
            saver.save(sess, path + 'model-' + str(episode_number) + '.cptk')


if __name__ == '__main__':
    # try:
    print("starting...")
    # reinforce_node()
    qlearning_node()
    # except rospy.ROSInterruptException:
    #     pass

