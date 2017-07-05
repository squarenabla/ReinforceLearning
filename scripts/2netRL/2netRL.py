#!/usr/bin/env python

import os
import sys
import threading
import atexit

import tensorflow as tf
import random
import numpy as np
import math

import datetime
import rospy

from rotors_reinforce.srv import PerformAction
from rotors_reinforce.srv import GetState

from History import *
from TFGraph import *
from TFLearningGraph import *
from ExperienceBuffer import *

#STATE_SIZE = 3 + 4 + 3 + 3
STATE_SIZE = 2 * 2
ACTION_NUM = 1
ACTION_SIZE = 4
TRAINING_SET_SIZE = 512


class Config:
    def __init__(self):
        self.batch_size = 512

        self.po_lrate = 0.1
        self.po_learning_rate_minimum = 0.001
        self.v_lrate = 0.5
        self.v_learning_rate_minimum = 0.01
        self.ql_lrate = 0.

        self.po_decay_step = 100
        self.v_decay_step = 100


class Statistics:
    def __init__(self, sess, logpath, scope='var'):
        self.scope = scope
        self.sess = sess

        self.writer = tf.summary.FileWriter(logpath)

        with tf.variable_scope('statistics'):
            self.values = tf.placeholder(tf.float32, [None], name=self.scope)
            #print self.episode_rewards.name
            self.variable_summaries(self.values)
            self.merged = tf.summary.merge_all()

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

        #print var.name
        with tf.variable_scope('summaries'):
            mean = tf.reduce_mean(var)
            #print var.name
            tf.summary.scalar('mean', mean)
            with tf.variable_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            tf.summary.scalar('sum', tf.reduce_sum(var))

    def add(self, data, step):
        statistics = self.sess.run(self.merged, feed_dict={self.values: data})
        self.writer.add_summary(statistics, step)
        self.writer.flush()


class GlobalStep:
    def __init__(self):
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

    def assign(self, sess, step):
        sess.run(self.step_assign_op, feed_dict={self.step_input: step})

    def getStep(self, sess):
        return sess.run(self.step_op)


global stop_flag
stop_flag = False

def termination_funk():
    print "terminatint\n"
    print "wainting for threads\n"

    stop_flag = True

    main_thread = threading.currentThread()

    for t in threading.enumerate():
        if t is not main_thread:
            t.join()

    save_path = saver.save(sess, path + 'model-final.cptk')
    print("Model saved in file: %s" % save_path)


atexit.register(termination_funk)

global path
path = "./tmp/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
if not os.path.exists(path):
    os.makedirs(path)

logpath = "./log/train/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
if not os.path.exists(logpath):
    os.makedirs(logpath)

def bufferFiller(sess, play_graph, train_graph, exp_buffer, graph_lock, statistics):
    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    history = History(ACTION_NUM, ACTION_SIZE)

    step = 0

    while not rospy.is_shutdown() and not stop_flag:
        rospy.wait_for_service('env_tr_get_state')
        try:
            response = get_state_client()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        height_state = np.concatenate(
            [[response.target_position[2] - response.position[2]], [response.position[2] - 0.0]])

        state = []

        for i in range(0, 2):
            state = np.concatenate(
                [state, [response.target_position[i] - response.position[i]], [response.position[i] - 0.0]])

        old_response = response

        history.clean()
        history.addState2History(state)

        total_reward = 0.0
        maxreward = 0.0


        action = np.zeros(1)

        while not rospy.is_shutdown():


            [height_action, _] = play_graph.calculateAction(sess, height_state)

            graph_lock.acquire()
            [xy_action, probability] = train_graph.calculateAction(sess, state)
            graph_lock.release()

            action[0] = np.random.choice(np.arange(0, ACTION_SIZE), p=probability[0][0])

            executed_action = np.zeros(4)

            if action[0] == 0:
                executed_action[0] = -1.0

            if action[0] == 1:
                executed_action[0] = 1.0

            if action[0] == 2:
                executed_action[1] = -1.0

            if action[0] == 3:
                executed_action[1] = 1.0

            # print executed_action
            executed_action[3] = float(height_action[0][0])

            # execute action
            rospy.wait_for_service('env_tr_perform_action')
            try:
                response = perform_action_client(executed_action)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                # print(response)

            height_state = np.concatenate([[response.target_position[2] - response.position[2]],
                                           [response.position[2] - old_response.position[2]]])

            state = []

            # distance to the target and velocity (3D)
            for i in range(0, 2):
                state = np.concatenate(
                    [state, [response.target_position[i] - response.position[i]],
                     [response.position[i] - old_response.position[i]]])

            old_response = response

            reward = response.reward
            done = response.crashed

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
            graph_lock.acquire()
            saver.save(sess, path + 'model-' + str(step) + '.cptk')
            graph_lock.release()
            maxreward = total_reward

        statistics.add([reward], step)

        graph_lock.acquire()
        predicted_rewards = train_graph.calculateRewards(sess, history.getStates())
        graph_lock.release()

        rewards = history.getReward()
        advantages = []
        update_vals = []

        for i, reward in enumerate(rewards):
            future_reward = 0
            future_transitions = len(rewards) - i
            decrease = 1
            for index2 in xrange(future_transitions):
                future_reward += rewards[(index2) + i]
                decrease = decrease * 0.97

            prediction = predicted_rewards[i]
            advantages.append(future_reward - prediction)
            update_vals.append(future_reward)

        exp_buffer.add(history.getStates(), history.getActions(), advantages, update_vals)


def teacher(sess, train_graph, exp_buffer, graph_lock, global_step):
    step = global_step.getStep(sess)

    while not stop_flag:
        if exp_buffer.getCompasity() < TRAINING_SET_SIZE:
            continue

        #update policy
        graph_lock.acquire()
        train_graph.updatePolicy(sess, exp_buffer, step)
        graph_lock.release()

        step += 1
        global_step.assign(step)


def reinforce_node():
    config = Config()

    play_graph = PlayGraph(config) #net1
    learn_graph = LearnGraph(config, scope='xy_plane') #net2
    global_step = GlobalStep()

    global sess
    sess = tf.Session()

    play_graph.constructGraph(sess, 2, 1, 2)
    learn_graph.constructGraph(sess, STATE_SIZE, ACTION_NUM, ACTION_SIZE)

    policy_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
    value_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')

    restore_vars = policy_var + value_var

    global restore
    restore = tf.train.Saver(var_list=restore_vars)
    #restore = tf.train.Saver()
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

    #bufferFiller(sess, play_graph, train_graph, buffer, graph_lock, statistics)
    exp_buffer = ExperienceBuffer()
    graph_lock = threading.Lock()
    statistics = Statistics(sess, logpath, scope="return")

    filler_thread = threading.Thread(target=bufferFiller, args=(sess,
                                                                play_graph,
                                                                learn_graph,
                                                                exp_buffer,
                                                                graph_lock,
                                                                statistics))

    teacher_thread = threading.Thread(target=teacher, args=(sess,
                                                            learn_graph,
                                                            exp_buffer,
                                                            graph_lock,
                                                            global_step))

    filler_thread.start()
    teacher_thread.start()


if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass

