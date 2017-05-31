import rospy
import sys
import math

import tensorflow as tf
import random

from rotors_reinforce.srv import *

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
            return {1,1,1,1,1,1}
        else:
            return {0,0,0,0,0,0}

    def updatePolicy(self, newstate):
        self.state = newstate

        #update the policy according to new state



def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    serviceClient = rospy.ServiceProxy('PerformAction', PerformAction)

    state = environmentState()

    #set up policy
    policy = Policy(state)

    # main loop:
    while not rospy.is_shutdown():

        #choose action according to policy
        action = policy.getAction()

        #execute action
        rospy.wait_for_service('ourService')
        try:
            response = serviceClient(action)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        #update environment
        state.currentPosition = response.position
        state.currentOrientation = response.orientation
        state.currentReward = response.reward
        #update policy
        policy.updatePolicy(state)

if __name__ == '__main__':
    try:
        reinforce_node()
    except rospy.ROSInterruptException:
        pass
 
