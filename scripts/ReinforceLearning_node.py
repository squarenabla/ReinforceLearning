#!/usr/bin/env python

import sys
import rospy
from rotors_reinforce.srv import PerformAction

import tensorflow as tf
import random
import numpy as np
import math

class environmentState:
    def __init__(self):
        self.currentPosition = {0,0,0}
        self.currentOrientation = {0,0,0,0}
        self.currentReward = 0

class Policy:
    def __init__(self, state):
        self.state = state
        #set up policy
        # Initialize table with all zeros
        self.Q = np.zeros([state_space, np.power(6, 6)])
        # Set learning parameters
        self.learningrate = .8
        self.discountrate = .95

    def getAction(self):
        i = 1 #TODO: replace with count
        a = np.argmax(self.Q[self.state.currentPosition, :] + np.random.randn(1, state_space) * (1 / (i + 1)))


    def updatePolicy(self, newstate, action):
        #update the policy according to new state
        # Update Q-Table with new knowledge
        self.Q[self.state.currentPosition, action] = self.Q[self.state.currentPosition, action] + self.learningrate * (newstate.currentReward + self.discountrate * np.max(self.Q[newstate.currentPosition, :]) - self.Q[self.state.currentPosition, action])
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
        action = policy.getAction()

        #execute action
        print(action)
        rospy.wait_for_service('env_tr_perform_action')
        try:
            response = serviceClient(action)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
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
 
