import tensorflow as tf
import numpy as np
import math

from ExperienceBuffer import *

class LearnGraph:
        def __init__(self, config, scope = ''):
            self.scope = scope

            self.batch_size = config.batch_size
            self.po_lrate = config.po_lrate
            self.po_learning_rate_minimum = config.po_learning_rate_minimum
            self.v_lrate = config.v_lrate
            self.v_learning_rate_minimum = config.v_learning_rate_minimum
            self.ql_lrate = config.ql_lrate

            self.global_step = tf.Variable(0, trainable=False)
            self.po_decay_step = config.po_decay_step
            self.v_decay_step = config.v_decay_step

        def constructGraph(self, sess, state_size, action_num, action_size):
            with tf.variable_scope(self.scope + "policy"):
                self.po_state = tf.placeholder(tf.float32, [None, state_size], name="po_state")
                self.po_prev_actions = tf.placeholder(tf.float32, [None, action_num, action_size], name="po_prev_action")
                self.po_return = tf.placeholder(tf.float32, [None], name="po_return")

                #formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                #hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2
                hid_neurons_num = 100

                self.po_W1 = tf.get_variable("policy_W1", [state_size, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b1 = tf.get_variable("policy_b1", [hid_neurons_num], initializer=tf.constant_initializer(0.0))

                self.po_W2 = tf.get_variable("policy_W2", [hid_neurons_num, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b2 = tf.get_variable("policy_b2", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                #
                self.po_W3 = tf.get_variable("policy_W3", [hid_neurons_num, action_num * action_size], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b3 = tf.get_variable("policy_b3", [action_num * action_size], initializer=tf.constant_initializer(0.0))

                self.po_output1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_state, self.po_W1), self.po_b1))
                self.po_output2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_output1, self.po_W2), self.po_b2))
                self.po_output3 = (tf.nn.bias_add(tf.matmul(self.po_output2, self.po_W3), self.po_b3))

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

            with tf.variable_scope(self.scope + "value"):
                self.v_state = tf.placeholder(tf.float32, [None, state_size], name="v_state")
                self.v_actual_return = tf.placeholder(tf.float32, [None], name="v_actual_return")

                # formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                #hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2
                hid_neurons_num = 100

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

        def calculateAction(self, sess, state):
            state_one_hot_sequence = np.expand_dims(state, axis = 0)
            action = sess.run(self.po_computed_actions, feed_dict={self.po_state : state_one_hot_sequence})
            probability = sess.run(self.po_probabilities, feed_dict={self.po_state : state_one_hot_sequence})
            return action, probability

        def calculateReward(self, sess, state):
            state_one_hot_sequence = np.expand_dims(state, axis=0)
            reward = sess.run(self.v_output3, feed_dict={self.v_state: state_one_hot_sequence})
            return reward[0][0]

        def calculateRewards(self, sess, states):
            rewards = sess.run(self.v_output3, feed_dict={self.v_state: states})
            return rewards[0]

        def updatePolicy(self, sess, buffer, step):
            states, actions, advantages, future_reward = buffer.sample(self.batch_size)

            _, v_lr = sess.run([self.v_optimizer, self.v_learning_rate_op], feed_dict={self.v_state: states,
                                                                                       self.v_actual_return: future_reward,
                                                                                       self.global_step: step
                                                                                       })

            _, W1, W3, p_lr = sess.run([self.po_optimizer, self.po_W1, self.po_W3, self.po_learning_rate_op], feed_dict={self.po_state: states,
                                                                                                                         self.po_prev_actions: actions,
                                                                                                                         self.po_return: advantages,
                                                                                                                         self.global_step: step
                                                                                                                         })
            return
