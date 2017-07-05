#!/usr/bin/env python

import os
import sys
import datetime
import rospy
from rotors_reinforce.srv import PerformAction
from rotors_reinforce.srv import GetState
from History import *

import gym
from gym import wrappers


import tensorflow as tf
import random
import numpy as np
import math


#STATE_SIZE = 3 + 4 + 3 + 3
STATE_SIZE = 2
ACTION_NUM = 1
ACTION_SIZE = 2

import atexit

def termination_funk():
    print "terminated\n"
    _train_writer.close()

atexit.register(termination_funk)

global path
path = "./tmp/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
if not os.path.exists(path):
    os.makedirs(path)

logpath = "./log/train/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
if not os.path.exists(logpath):
    os.makedirs(logpath)

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
            self.po_lrate = 0.02
            self.po_learning_rate_minimum = 0.005
            self.v_lrate = 0.1
            self.v_learning_rate_minimum = 0.03
            self.ql_lrate = 0.

            self.global_step = tf.Variable(0, trainable=False)
            self.po_decay_step = 50
            self.v_decay_step = 50

        def constructGraph(self, sess, state_size, action_num, action_size):
            with tf.variable_scope("policy"):
                self.episode_rewards = tf.placeholder(tf.float32, [None], name="episode_rewards")
                self.po_state = tf.placeholder(tf.float32, [None, state_size], name="po_state")
                self.po_prev_actions = tf.placeholder(tf.float32, [None, action_num, action_size], name="po_prev_action")
                self.po_return = tf.placeholder(tf.float32, [None], name="po_return")

                #formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2
                #hid_neurons_num = 32

                self.po_W1 = tf.get_variable("policy_W1", [state_size, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b1 = tf.get_variable("policy_b1", [hid_neurons_num], initializer=tf.constant_initializer(0.0))

                self.po_W2 = tf.get_variable("policy_W2", [hid_neurons_num, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b2 = tf.get_variable("policy_b2", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                #
                self.po_W3 = tf.get_variable("policy_W3", [hid_neurons_num, action_num * action_size], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b3 = tf.get_variable("policy_b3", [action_num * action_size], initializer=tf.constant_initializer(0.0))

                self.po_output1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_state, self.po_W1), self.po_b1))
                self.po_output2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_output1, self.po_W2), self.po_b2))
                self.po_output3 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_output2, self.po_W3), self.po_b3))

                self.po_probabilities = tf.nn.softmax(tf.reshape(self.po_output3, [-1, action_num, action_size]))
                self.po_max_probabilities = tf.reduce_max(self.po_probabilities, axis = -1)
                self.po_computed_actions = tf.argmax(self.po_probabilities, axis = -1)

                #self.po_matr_prev_actions = tf.reshape(self.po_prev_actions, [-1, action_num, action_size])
                self.po_eligibility = tf.log(tf.reduce_sum(tf.multiply(self.po_prev_actions, self.po_probabilities), -1))
                self.po_loss = tf.multiply(-tf.reduce_sum(self.po_eligibility, -1), self.po_return)



                self.po_learning_rate_op = tf.maximum(self.po_learning_rate_minimum,
                                                   tf.train.exponential_decay(
                                                       self.po_lrate,
                                                       self.global_step,
                                                       self.po_decay_step,
                                                       0.96,
                                                       staircase=True))

                self.po_optimizer = tf.train.AdamOptimizer(self.po_learning_rate_op).minimize(self.po_loss)

            with tf.variable_scope("value"):
                self.v_state = tf.placeholder(tf.float32, [None, state_size], name="v_state")
                self.v_actual_return = tf.placeholder(tf.float32, [None, 1], name="v_actual_return")

                # formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2
                #hid_neurons_num = 32

                self.v_W1 = tf.get_variable("value_V1", [state_size, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.v_b1 = tf.get_variable("value_b1", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                self.v_W2 = tf.get_variable("value_W2", [hid_neurons_num, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.v_b2 = tf.get_variable("value_b2", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                self.v_W3 = tf.get_variable("value_W3", [hid_neurons_num, 1], initializer=tf.contrib.layers.xavier_initializer())
                self.v_b3 = tf.get_variable("value_b3", [1], initializer=tf.constant_initializer(0.0))

                # in the future, try to calculate return for for each action

                self.v_output1 = tf.nn.bias_add(tf.matmul(self.v_state, self.v_W1), self.v_b1)
                self.v_output2 = tf.nn.bias_add(tf.matmul(self.v_output1, self.v_W2), self.v_b2)
                self.v_output3 = tf.nn.bias_add(tf.matmul(self.v_output2, self.v_W3), self.v_b3)

                self.v_diffs = self.v_output3 - self.v_actual_return
                self.v_loss = tf.nn.l2_loss(self.v_diffs)

                self.v_learning_rate_op = tf.maximum(self.v_learning_rate_minimum,
                                                   tf.train.exponential_decay(
                                                       self.v_lrate,
                                                       self.global_step,
                                                       self.v_decay_step,
                                                       0.96,
                                                       staircase=True))

                self.v_optimizer = tf.train.AdamOptimizer(self.v_learning_rate_op).minimize(self.v_loss)

            self.constructSummary(sess)

        def constructSummary(self, sess):
            variable_summaries(self.episode_rewards)
            self.merged = tf.summary.merge_all()
            global _train_writer
            _train_writer = tf.summary.FileWriter(logpath, sess.graph)

        def calculateAction(self, sess, state):
            state_one_hot_sequence = np.expand_dims(state, axis = 0)
            action = sess.run(self.po_computed_actions, feed_dict={self.po_state : state_one_hot_sequence})
            probability = sess.run(self.po_max_probabilities, feed_dict={self.po_state : state_one_hot_sequence})
            return action, probability

        def calculateReward(self, sess, state):
            state_one_hot_sequence = np.expand_dims(state, axis=0)
            reward = sess.run(self.v_output3, feed_dict={self.v_state: state_one_hot_sequence})
            return reward[0][0]

        def updatePolicy(self, sess, history, step):
            rewards = history.getRewardHistory()
            advantages = []
            update_vals = []

            for i, reward in enumerate(rewards):
                future_reward = 0
                future_transitions = len(rewards) - i
                decrease = 1
                for index2 in xrange(future_transitions):
                    future_reward += rewards[(index2) + i]
                    decrease = decrease * 0.97

                prediction = self.calculateReward(sess, history.getState(i))
                advantages.append(future_reward - prediction)
                update_vals.append(future_reward)


            sess.run(self.v_optimizer, feed_dict={self.v_state: history.getStates(),
                                                  self.v_actual_return: np.expand_dims(update_vals, axis=1),
                                                  self.global_step: step
                                                  })

            statistics, _, W1, W3 = sess.run([self.merged, self.po_optimizer, self.po_W1, self.po_W3], feed_dict={self.po_state: history.getStates(),
                                                                                                                  self.po_prev_actions: history.getActions(),
                                                                                                                  self.po_return: advantages,
                                                                                                                  self.episode_rewards: history.getRewardHistory(),
                                                                                                                  self.global_step: step
                                                                                                                  })
            _train_writer.add_summary(statistics, step)
            _train_writer.flush()

            #print W1
            #print W3



def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    history = History(ACTION_NUM, ACTION_SIZE)

    graph = ComputationalGraph()

    action = np.zeros(ACTION_NUM)
    executed_action = np.zeros(3 + ACTION_NUM)

    global sess
    sess = tf.Session()

    graph.constructGraph(sess, STATE_SIZE, ACTION_NUM, ACTION_SIZE)

    global saver
    saver = tf.train.Saver()



    try:
        saver.restore(sess, "./Graph/model-466.cptk")
        print "model restored"
    except:
        print "model isn't restored. random initialization"
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)


    startE = 1.0
    endE = 0.0
    ep = startE
    #anneling_steps = buffer_size
    anneling_steps = 800
    stepDrop = (startE - endE) / anneling_steps

    total_step = 0

    # main loop:
    while not rospy.is_shutdown():
        crashed_flag = False

        #print "new episode"
        #get initial state
        #print "Get initial state"
        rospy.wait_for_service('env_tr_get_state')
        try:
            response = get_state_client()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        #state = env.reset()
        old_position = 0

        state = np.concatenate([[response.target_position[2] - response.position[2]], [response.position[2] - old_position]])
        old_position = response.position[2]

        history.clean()
        history.addState2History(state)

        step = 0
        total_reward = 0

        # run episode, while not crashed and simulation is running
        while 1 and not rospy.is_shutdown():
            #get most probable variant to act for each action, and the probabilities

            [computed_action, probability] = graph.calculateAction(sess, history.getLastState())

            #choose action stochasticaly
            for i, act in enumerate(computed_action[0]):
                action[i] = int(computed_action[0][i])


            executed_action[3] = float(action[0]) / float(ACTION_SIZE - 1)

            #execute action
            #print(action)
            #print(executed_action)
            rospy.wait_for_service('env_tr_perform_action')
            try:
                response = perform_action_client(executed_action)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            #print(response)

            state = np.concatenate([[response.target_position[2] - response.position[2]], [response.position[2] - old_position]])
            old_position = response.position[2]

            reward = response.reward
            done = response.crashed

            #update history
            history.addAction2History(action)
            history.addState2History(state)
            history.addReward2History(reward)

            crashed_flag = done

            step = step + 1
            total_reward += reward

        print total_step, total_reward
        total_step = total_step + 1
        ep -= stepDrop

            #rewardList.append(total_reward)


if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass
