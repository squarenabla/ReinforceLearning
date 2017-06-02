#!/usr/bin/env python

import sys
import rospy
from rotors_reinforce.srv import PerformAction

import tensorflow as tf
import random
import numpy as np
import math


class ComputationalGraph:
    def __init__(self):
        self.po_lrate = 0.02

    def constructGraph(self, state_size, action_num, action_size):
        with tf.variable_scope("policy"):
            self.po_W1 = tf.get_variable("policy_W1", [state_size, state_size])
            self.b_1 = tf.get_variable("policy_b1", [state_size])

            self.po_W2 = tf.get_variable("policy_W2", [state_size, action_num * action_size])
            self.b_2 = tf.get_variable("policy_b2", [action_num * action_size])

            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.prev_actions = tf.placeholder(tf.float32, [None, action_num * action_size])
            self.returnn = tf.placeholder(tf.float32, [None, 1])

            self.output1 = tf.nn.sigmoid(tf.add(tf.matmul(self.state, self.po_W1), self.b_1))
            self.output2 = tf.add(tf.matmul(self.output1, self.po_W2), self.b_2)

            self.probabilities = tf.nn.softmax(tf.reshape(self.output2, [-1, action_num, action_size]))


            self.matr_prev_actions = tf.reshape(self.prev_actions, [-1, action_num, action_size])
            self.eligibility = tf.log(tf.reduce_sum(tf.multiply(self.matr_prev_actions, self.probabilities), -1)) * self.returnn
            self.loss = -tf.reduce_sum(self.eligibility, -1)
            self.po_optimizer = tf.train.AdamOptimizer(self.po_lrate).minimize(self.loss)


class environmentState:
    def __init__(self):
        self.currentPosition = [0,0,0]
        self.currentOrientation = [0,0,0,0]
        self.currentReward = 0





class Policy:
    def __init__(self, state):
        self.state = state
        #set up policy
        # Initialize table with all zeros
        self.Q = np.zeros([np.power(3, 10), 4*4])
        # Set learning parameters
        self.learningrate = .8
        self.discountrate = .95

    def getAction(self):



    def updatePolicy(self, newstate, action):
        #update the policy according to new state
        # Update Q-Table with new knowledge

        self.state = newstate



def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    serviceClient = rospy.ServiceProxy('env_tr_perform_action', PerformAction)

    state = environmentState()

    #initialize policy
    policy = Policy(state)

    # create lists to contain total rewards and steps per episode
    #rewardList = []

    # main loop:
    while not rospy.is_shutdown(): #TODO: track episodes

        #choose action according to policy
        #action = policy.getAction()
        if (state.currentPosition[2] < 5):
            action = [0,0,0,1]
        else:
            action = [0,0,0,0]

        #execute action
        print("action:")
        print(action)
        rospy.wait_for_service('env_tr_perform_action')
        try:
            response = serviceClient(action)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        print("response:")
        print(response)

        #update environment
        state.currentPosition = response.position
        state.currentOrientation = response.orientation
        state.currentReward = response.reward

        #update policy
        policy.updatePolicy(state, action)

        #rewardList.append(total_reward)

if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass
 
