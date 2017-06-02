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
        self.currentPosition = [0,0,0]
        self.currentOrientation = [0,0,0,0]
        self.currentReward = 0

    def get(self):
        result = [0, 0, 0]

        i = 1

        while (i < 11):
            if (self.currentPosition[0] < i):
                result[0] = result[0] + 1
            if (self.currentPosition[1] < i):
                result[1] = result[1] + 1
            if (self.currentPosition[2] < i):
                result[2] = result[2] + 1
            i = i+1

        return result



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
        i = 1 #TODO: replace with count
        print(self.state.get())
        print(self.Q[self.state.get(), :])
        a = np.argmax(self.Q[self.state.get(), :] + np.random.randn(3, 4) * (1 / (i + 1)))
        return a


    def updatePolicy(self, newstate, action):
        #update the policy according to new state
        # Update Q-Table with new knowledge
        self.Q[self.state.get(), action] = self.Q[self.state.get(), action] + self.learningrate * (newstate.currentReward + self.discountrate * np.max(self.Q[newstate.get(), :]) - self.Q[self.state.get(), action])
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
        #policy.updatePolicy(state, action)

        #rewardList.append(total_reward)

if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass
 
