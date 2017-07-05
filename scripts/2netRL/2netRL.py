#!/usr/bin/env python

import os
import sys
import datetime
import rospy
from rotors_reinforce.srv import PerformAction
from rotors_reinforce.srv import GetState
from History import *
from TFGraph import *

import gym
from gym import wrappers

# env = gym.make('CartPole-v0')
# env.render()
# monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)


import tensorflow as tf
import random
import numpy as np
import math


#STATE_SIZE = 3 + 4 + 3 + 3
STATE_SIZE = 2 * 2
ACTION_NUM = 1
ACTION_SIZE = 4

import atexit

def termination_funk():
    print "terminated\n"
    _train_writer.close()

    save_path = saver.save(sess, path + 'model-final.cptk')
    print("Model saved in file: %s" % save_path)


class Config:
    def __init__(self):
        self.po_lrate = 0.04
        self.po_learning_rate_minimum = 0.001
        self.v_lrate = 0.2
        self.v_learning_rate_minimum = 0.01
        self.ql_lrate = 0.

        self.po_decay_step = 20
        self.v_decay_step = 10

atexit.register(termination_funk)

global path
path = "./tmp/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
if not os.path.exists(path):
    os.makedirs(path)

logpath = "./log/train/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
if not os.path.exists(logpath):
    os.makedirs(logpath)

config = Config()

def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    history = History(ACTION_NUM, ACTION_SIZE)

    height_graph = ComputationalGraph(config) #net1
    xy_plane_graph = ComputationalGraph(config, scope='xy_plane') #net2

    action = np.zeros(1)



    global sess
    sess = tf.Session()

    global _train_writer
    _train_writer = tf.summary.FileWriter(logpath, sess.graph)

    with tf.variable_scope('step'):
        step_op = tf.Variable(0, trainable=False, name='step')
        step_input = tf.placeholder('int32', None, name='step_input')
        step_assign_op = step_op.assign(step_input)

    height_graph.constructGraph(sess, 2, 1, 2)
    xy_plane_graph.constructGraph(sess, STATE_SIZE, ACTION_NUM, ACTION_SIZE)

    policy_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
    value_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')

    restore_vars = policy_var + value_var

    global restore
    #restore = tf.train.Saver(var_list=restore_vars)
    restore = tf.train.Saver()
    global saver
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    try:
        restore.restore(sess, "./Graph/model-final.cptk")
        print "model restored"
    except:
        print "model isn't restored. random initialization"
        init_op = tf.global_variables_initializer()
        sess.run(init_op)




    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    maxreward = 0


    startE = 1.0
    endE = 0.0
    ep = startE
    #anneling_steps = buffer_size
    anneling_steps = 800
    stepDrop = (startE - endE) / anneling_steps

    total_step = sess.run(step_op)
    print total_step

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
        old_position = [0.0, 0.0, 0.0]


        #state for net1
        height_state = np.concatenate([[response.target_position[2] - response.position[2]], [response.position[2] - 0.0]])

        #state for net2
        state = []
        #distance to the target and velocity (3D)
        for i in range(0,2):
            state = np.concatenate([state, [response.target_position[i] - response.position[i]], [response.position[i] - 0.0]])

        #angular position and angular velocity (3D)
        # for i in range(0, 3):
        #     state = np.concatenate([state, [float(response.orientation[i])/float(response.orientation[3])]])
        old_response = response

        history.clean()
        history.addState2History(state)

        step = 0
        total_reward = 0

        # run episode, while not crashed and simulation is running
        while 1 and not rospy.is_shutdown():
            #get most probable variant to act for each action, and the probabilities

            #[computed_action, probability] = graph.calculateAction(sess, history.getLastState())

            [height_action, _] = height_graph.calculateAction(sess, height_state)
            [xy_action, probability] = xy_plane_graph.calculateAction(sess, state)

            #choose action stochasticaly
            #print probability
            action[0] = np.random.choice(np.arange(0,ACTION_SIZE), p=probability[0][0])
            # if random.uniform(0, 1) > probability[0][0]:
            #     action[0] = np.random.randint(ACTION_SIZE)
            # else:
            #     action[0] = int(xy_action[0][0])

            executed_action = np.zeros(4)

            if  action[0] == 0:
                executed_action[0] = -1.0

            if  action[0] == 1:
                executed_action[0] = 1.0

            if  action[0] == 2:
                executed_action[1] = -1.0

            if  action[0] == 3:
                executed_action[1] = 1.0

            # if  action[0] == 5:
            #     executed_action[2] = -1.0
            #
            # if  action[0] == 6:
            #     executed_action[2] = 1.0

            #print executed_action
            executed_action[3] = float(height_action[0][0])

            #execute action
            #print(action)
            #print(executed_action)
            rospy.wait_for_service('env_tr_perform_action')
            try:
                response = perform_action_client(executed_action)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            #print(response)


            # state = []
            # for i in range(0, 3):
            #     state = np.concatenate([state, [response.target_position[i] - response.position[i]],
            #                             [response.position[i] - old_position[i]]])
            # old_position = response.position


            height_state = np.concatenate([[response.target_position[2] - response.position[2]], [response.position[2] - old_response.position[2]]])

            state = []

            # distance to the target and velocity (3D)
            for i in range(0, 2):
                state = np.concatenate(
                    [state, [response.target_position[i] - response.position[i]], [response.position[i] - old_response.position[i]]])

            # angular position and angular velocity (3D)
            # for i in range(0, 3):
            #     state = np.concatenate([state, [float(response.orientation[i]) / float(response.orientation[3])]])
            old_response = response

            reward = response.reward
            done = response.crashed

            #update history
            history.addAction2History(action)
            history.addState2History(state)
            history.addReward2History(reward)

            crashed_flag = done

            step = step + 1
            total_reward += reward


            if crashed_flag:
                break

        if total_reward > maxreward:
            print "saving model"
            saver.save(sess, path + 'model-' + str(total_step) + '.cptk')
            maxreward = total_reward
            #break
        #update policy
        statistics = xy_plane_graph.updatePolicy(sess, history, total_step)

        _train_writer.add_summary(statistics, total_step)
        _train_writer.flush()

        print total_step, total_reward
        total_step = total_step + 1
        sess.run(step_assign_op, feed_dict={step_input: total_step})
        ep -= stepDrop

            #rewardList.append(total_reward)


if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass

