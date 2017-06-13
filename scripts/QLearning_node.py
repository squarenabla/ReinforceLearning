#!/usr/bin/env python

import sys
import rospy
from rotors_reinforce.srv import PerformAction
from rotors_reinforce.srv import GetState

import tensorflow as tf
import random
import numpy as np
import math


#STATE_SIZE = 3 + 4 + 3 + 3
STATE_SIZE = 1 + 1
ACTION_NUM = 1
ACTION_SIZE = 3

import atexit

def termination_funk():
    print "terminated\n"
    _train_writer.close()

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)


atexit.register(termination_funk)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    tf.summary.scalar('sum', tf.reduce_sum(var))

class ComputationalGraph:
        def __init__(self):
            self.po_lrate = 0.01
            self.v_lrate = 0.1
            self.ql_lrate = 0.1

        def constructGraph(self, sess, state_size, action_num, action_size):

            with tf.variable_scope("Q-learning"):
                self.ql_state = tf.placeholder(tf.float32, [1, state_size], name="ql_state")

                # formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2

                self.ql_W1 = tf.get_variable("ql_W1", [state_size, action_num * action_size])
                #self.ql_W2 = tf.Variable("ql_W2", [hid_neurons_num, action_num * action_size])

                #self.ql_output1 = tf.matmul(self.ql_state, self.ql_W1)
                self.ql_Q = tf.matmul(self.ql_state, self.ql_W1)
                self.ql_computated_action = tf.argmax(self.ql_Q, axis=-1)

                self.ql_Qnext = tf.placeholder(tf.float32, [1,action_num * action_size], name="ql_Qnext")
                self.ql_loss = tf.reduce_sum(tf.square(self.ql_Qnext - self.ql_Q))
                self.ql_optimizer = tf.train.AdamOptimizer(self.ql_lrate).minimize(self.ql_loss)

            with tf.variable_scope("summary"):
                self.episode_rewards = tf.placeholder(tf.float32, [None], name="episode_rewards")

            self.constructSummary(sess)

        def constructSummary(self, sess):
            variable_summaries(self.episode_rewards)
            self.merged = tf.summary.merge_all()
            global _train_writer
            _train_writer = tf.summary.FileWriter('./log/train', sess.graph)

        def qLearningPrediction(self, sess, state):
            state_2D = np.expand_dims(state, axis=0)
            action, Q = sess.run([self.ql_computated_action, self.ql_Q], feed_dict={self.ql_state: state_2D})
            return action, Q

        def getQ(self, sess, state):
            state_2D = np.expand_dims(state, axis=0)
            Q = sess.run(self.ql_Q, feed_dict={self.ql_state: state_2D})
            return Q

        def optimizeQ(self, sess, state, targetQ):
            state_2D = np.expand_dims(state, axis=0)
            _, W = sess.run([self.ql_optimizer, self.ql_W1], feed_dict={self.ql_state: state_2D,
                                                                        self.ql_Qnext: targetQ})

            print W #print current weights of network


        def addReward2Summary(self, sess, reward, step):
            statistics = sess.run(self.merged, feed_dict={self.episode_rewards: reward})
            _train_writer.add_summary(statistics, step)
            _train_writer.flush()

def qlearning_node():
#set up env
    rospy.init_node('qLearning', anonymous=True)
    perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    graph = ComputationalGraph()

    action = np.zeros(ACTION_NUM)
    executed_action = np.zeros(3 + ACTION_NUM)

    global sess
    sess = tf.Session()

    graph.constructGraph(sess, STATE_SIZE, ACTION_NUM, ACTION_SIZE)

    global saver
    saver = tf.train.Saver()

    try:
        saver.restore(sess, "./tmp/model.ckpt")
        print "model restored"
    except:
        print "model isn't restored. random initialization"
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    step = 0
    maxstep = 100
    e = 0.1
    gamma = 0.99

    episode_number = 0
    total_reward = np.zeros(1)

    # main loop:
    while not rospy.is_shutdown():
        crashed_flag = False

        print "new episode"
        #get initial state
        print "Get initial state"
        rospy.wait_for_service('env_tr_get_state')
        try:
            response = get_state_client()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e



        # run episode, while not crashed and simulation is running
        while not crashed_flag and not rospy.is_shutdown():
            #get most probable variant to act for each action, and the probabilities

            action, Q = graph.qLearningPrediction(sess, np.concatenate([response.position[2:], response.target_position[2:]]))

            if np.random.rand(1) < e:
                action[0] = np.random.randint(ACTION_SIZE)
            else:
                action[0] = int(action[0])

            executed_action[3] = float(action[0]) / float(ACTION_SIZE - 1)

            #print(executed_action)
            rospy.wait_for_service('env_tr_perform_action')
            try:
                new_response = perform_action_client(executed_action)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
            #print(new_response)

            Q1 = graph.getQ(sess, np.concatenate([new_response.position[2:], new_response.target_position[2:]]))

            targetQ = Q
            targetQ[0, action[0]] = response.reward + gamma * np.max(Q1)
            total_reward[0] = total_reward[0] + response.reward

            graph.optimizeQ(sess, np.concatenate([response.position[2:], response.target_position[2:]]), targetQ)

            response = new_response

            crashed_flag = response.crashed

            step = step + 1



        maxstep = max(maxstep, step)
        e = 1.0/((2 * step/(maxstep) + 10))
        step = 0

        graph.addReward2Summary(sess, total_reward, episode_number)
        episode_number = episode_number + 1
        total_reward[0] = 0

            #rewardList.append(total_reward)


if __name__ == '__main__':
    try:
        print("starting...")
        # reinforce_node()
        qlearning_node()
    except rospy.ROSInterruptException:
        pass

