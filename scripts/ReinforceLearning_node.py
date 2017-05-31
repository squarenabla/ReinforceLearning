#!/usr/bin/env python

import sys
import rospy
import math
from rotors_reinforce.srv import PerformAction

import tensorflow as tf
import random

class environmentState:
    def __init__(self):
        self.currentPosition = {0,0,0}
        self.currentOrientation = {0,0,0,0}
        self.currentReward = 0

class Policy:
    def __init__(self, state):
        self.state = state
        #initialize policy

    def getAction(self):
        #random policy
        if (random.random() < 0.8):
            return [1,1,1,1,1,1]
        else:
            return [0,0,0,0,0,0]

    def updatePolicy(self, newstate):
        self.state = newstate
        #update the policy according to new state



def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    serviceClient = rospy.ServiceProxy('env_tr_perform_action', PerformAction)

    state = environmentState()

    #set up policy
    policy = Policy(state)

    # main loop:
    while not rospy.is_shutdown():

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
        policy.updatePolicy(state)

if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass
 
