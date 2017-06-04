#!/usr/bin/env python

import sys
import rospy
from rotors_reinforce.srv import PerformAction
from rotors_reinforce.srv import GetState

import tensorflow as tf
import random
import numpy as np
import math


STATE_SIZE = 3 + 4 + 3
ACTION_NUM = 4
ACTION_SIZE = 3

class ComputationalGraph:
    def __init__(self):
        self.po_lrate = 0.02
        self.v_lrate = 0.1

    def constructGraph(self, state_size, action_num, action_size):
        with tf.variable_scope("policy"):
            self.po_W1 = tf.get_variable("policy_W1", [state_size, state_size])
            self.b_1 = tf.get_variable("policy_b1", [state_size])

            self.po_W2 = tf.get_variable("policy_W2", [state_size, action_num * action_size])
            self.b_2 = tf.get_variable("policy_b2", [action_num * action_size])

            self.po_state = tf.placeholder(tf.float32, [None, state_size])
            self.po_prev_actions = tf.placeholder(tf.float32, [None, action_num * action_size])
            self.po_return = tf.placeholder(tf.float32, [None, 1])

            self.po_output1 = tf.nn.sigmoid(tf.add(tf.matmul(self.po_state, self.po_W1), self.b_1))
            self.po_output2 = tf.add(tf.matmul(self.po_output1, self.po_W2), self.b_2)

            self.po_probabilities = tf.nn.softmax(tf.reshape(self.po_output2, [-1, action_num, action_size]))
            self.po_max_probabilities = tf.reduce_max(self.po_probabilities, axis = -1)
            self.po_computed_actions = tf.argmax(self.po_probabilities, axis = -1)

            self.po_matr_prev_actions = tf.reshape(self.po_prev_actions, [-1, action_num, action_size])
            self.po_eligibility = tf.log(tf.reduce_sum(tf.multiply(self.po_matr_prev_actions, self.po_probabilities), -1)) * self.po_return
            self.po_loss = -tf.reduce_sum(self.po_eligibility, -1)
            self.po_optimizer = tf.train.AdamOptimizer(self.po_lrate).minimize(self.po_loss)

        with tf.variable_scope("value"):
            self.v_state = tf.placeholder(tf.float32, [None, state_size])
            self.v_actual_return = tf.placeholder(tf.float32, [None, 1])


            self.v_W1 = tf.get_variable("value_V1", [state_size, 10])
            self.v_b1 = tf.get_variable("value_b1", [10])
            self.v_W2 = tf.get_variable("value_W2", [10, 1])
            self.v_b2 = tf.get_variable("value_b2", [1])
            # in the future, try to calculate return for for each action

            self.v_output1 = tf.nn.sigmoid(tf.matmul(self.v_state, self.v_W1) + self.v_b1)
            self.v_output2 = tf.matmul(self.v_output1, self.v_W2) + self.v_b2
            self.v_diffs = self.v_output2 - self.v_actual_return
            self.v_loss = tf.nn.l2_loss(self.v_diffs)
            self.v_optimizer = tf.train.AdamOptimizer(self.v_lrate).minimize(self.v_loss)

    def calculateAction(self, sess, state):
        state_one_hot_sequence = np.expand_dims(state, axis = 0)
        action = sess.run(self.po_computed_actions, feed_dict={self.po_state : state_one_hot_sequence})
        probability = sess.run(self.self.po_max_probabilities, feed_dict={self.po_state : state_one_hot_sequence})
        return action, probability

    def calculateReward(self, sess, state):
        state_one_hot_sequence = np.expand_dims(state, axis=0)
        reward = sess.run(self.v_output2, feed_dict={self.v_state: state_one_hot_sequence})
        return reward[0][0]

    def updatePolicy(self, sess, history):
        rewards = history.getRewardHistory()
        advantages = []
        update_vals = []

        for i, reward in enumerate(rewards):
            future_reward = 0
            future_transitions = len(rewards) - i
            decrease = 1
            for index2 in xrange(future_transitions):
                future_reward += rewards[(index2) + i] * decrease
                decrease = decrease * 0.97

            prediction = self.calculateReward(sess, history.getState(i))
            advantages.append(future_reward - prediction)
            update_vals.append(future_reward)

        assert (len(history.getActions()) == len(history.getStates()) == len(advantages)), "Size of action, state and reward arrays don't match"

        sess.run(self.v_optimizer, feed_dict={self.v_state: history.getStates()})
        sess.run(self.po_optimizer, feed_dict={self.po_state: history.getStates(),
                                               self.po_prev_actions: history.getActions(),
                                               self.po_return: np.expand_dims(advantages, axis=1)})



class History:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        #examination is needed, whether to use return or not
        #self.returnn = []

    def addAction2History(self, action):
        action_matrix = np.zeros((ACTION_NUM, ACTION_SIZE), dtype = np.float32)
        for i, act in enumerate(action):
            action_matrix[i][act] = 1.0
        self.actions.append(action_matrix)

    def getActions(self):
        return self.actions

    def getLastAction(self):
        assert (len(self.actions) > 0), "Action history is empty!"
        return self.actions[-1]

    def addState2History(self, response):
        state = np.concatenate([response.currentPosition, response.orientation, response.target_position])
        self.states.append(state)

    def getStates(self):
        return self.states[:-1]

    def getState(self, iterator):
        assert (len(self.states) > 0), "State history is empty!"
        return self.states[iterator]

    def getLastState(self):
        assert (len(self.states) > 0), "State history is empty!"
        return self.states[-1]

    def addReward2History(self, reward):
        self.rewards.append(reward)

    def getRewardHistory(self):
        return self.rewards

    def getLastReward(self):
        assert (len(self.rewards) > 0), "Reward history is empty!"
        return self.rewards[-1]

##############
##OLD CLASSES#
##############

# class environmentState:
#     def __init__(self):
#         self.currentPosition = np.array([0.0, 0.0, 0.0])
#         self.currentOrientation = np.array([0.0, 0.0, 0.0, 0.0])
#         self.currentReward = 0.0
#
# class Policy:
#     def __init__(self, state):
#         self.state = state
#         #set up policy
#         # Initialize table with all zeros
#         self.Q = np.zeros([state_space, np.power(6, 6)])
#         # Set learning parameters
#         self.learningrate = .8
#         self.discountrate = .95
#
#     def getAction(self):
#         i = 1 #TODO: replace with count
#         a = np.argmax(self.Q[self.state.currentPosition, :] + np.random.randn(1, state_space) * (1 / (i + 1)))
#
#
#     def updatePolicy(self, newstate, action):
#         #update the policy according to new state
#         # Update Q-Table with new knowledge
#         self.Q[self.state.currentPosition, action] = self.Q[self.state.currentPosition, action] + self.learningrate * (newstate.currentReward + self.discountrate * np.max(self.Q[newstate.currentPosition, :]) - self.Q[self.state.currentPosition, action])
#         self.state = newstate

##############
##OLD CLASSES#
##############



def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    history = History()

    graph = ComputationalGraph()
    graph.constructGraph(STATE_SIZE, ACTION_NUM, ACTION_SIZE)
    init_op = tf.global_variables_initializer()

    executed_action = np.zeros(ACTION_NUM)

    with tf.Session() as sess:
        sess.run(init_op)
        # main loop:
        while not rospy.is_shutdown():
            crashed_flag = False

            #get initial state
            print "Get initial state"
            rospy.wait_for_service('env_tr_get_state')
            try:
                response = get_state_client()
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            history.addState2History(response)

            # run episode, while not crashed and simulation is running
            while not crashed_flag and not rospy.is_shutdown():
                #get most probable variant to act for each action, and the probabilities
                [computed_action, probability] = graph.calculateAction(sess, history.getLastState())

                #choose action stochasticaly
                for i, act in enumerate(computed_action):
                    if random.uniform(0, 1) > probability[0][i]:
                        executed_action[i] = np.random.randint(ACTION_SIZE)
                    else:
                        executed_action[i] = computed_action[0][i]

                #execute action
                print(executed_action)
                rospy.wait_for_service('env_tr_perform_action')
                try:
                    response = perform_action_client(executed_action)
                except rospy.ServiceException, e:
                    print "Service call failed: %s" % e
                print(response)

                #update history
                history.addAction2History(executed_action)
                history.addState2History(response)
                history.addReward2History(response.reward)

                crashed_flag = response.crashed


            #update policy
            graph.updatePolicy(history)

            #rewardList.append(total_reward)

if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass
 
