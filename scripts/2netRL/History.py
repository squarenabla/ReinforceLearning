#!/usr/bin/env python

import numpy as np

class History:
    def __init__(self, actionNumber, actionSize):
        self.states = []
        self.actions = []
        self.rewards = []
        self.actionNumber = actionNumber
        self.actionSize = actionSize
        #examination is needed, whether to use return or not
        #self.returnn = []

    def addAction2History(self, action):
        action_matrix = np.zeros((self.actionNumber, self.actionSize), dtype = np.float32)
        for i, act in enumerate(action):
            #print action[i], act
            action_matrix[i][int(act)] = 1.0
        self.actions.append(action_matrix)
        #self.actions.append(action)

    def getActions(self):
        return self.actions

    def getLastAction(self):
        assert (len(self.actions) > 0), "Action history is empty!"
        return self.actions[-1]

    def addState2History(self, state):
        #state = np.concatenate([response.position, response.velocity, response.target_position, response.orientation])
        #state = np.concatenate([response.position[2:], response.target_position[2:]])
        # state = response
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

    def sumRewards(self):
        return sum(self.rewards)

    def clean(self):
        self.states = []
        self.actions = []
        self.rewards = []