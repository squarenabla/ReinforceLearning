import tensorflow as tf
import numpy as np
import math



class ComputationalGraph:
        def __init__(self, config, scope = ''):
            self.scope = scope

            self.config = config
            self.po_lrate = config.po_lrate
            self.po_learning_rate_minimum = config.po_learning_rate_minimum
            self.v_lrate = config.v_lrate
            self.v_learning_rate_minimum = config.v_learning_rate_minimum
            self.ql_lrate = config.ql_lrate

            self.global_step = tf.Variable(0, trainable=False)
            self.po_decay_step = config.po_decay_step
            self.v_decay_step = config.v_decay_step

        def constructGraph(self, sess, state_size, action_num, action_size):
            with tf.variable_scope(self.scope + "policy"):
                self.po_state = tf.placeholder(tf.float32, [None, state_size], name="po_state")
                self.po_prev_actions = tf.placeholder(tf.float32, [None, action_num, action_size], name="po_prev_action")
                self.po_return = tf.placeholder(tf.float32, [None], name="po_return")

                #formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                #hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2
                hid_neurons_num = 50

                self.po_W1 = tf.get_variable("policy_W1", [state_size, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b1 = tf.get_variable("policy_b1", [hid_neurons_num], initializer=tf.constant_initializer(0.0))

                self.po_W2 = tf.get_variable("policy_W2", [hid_neurons_num, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b2 = tf.get_variable("policy_b2", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                #
                self.po_W3 = tf.get_variable("policy_W3", [hid_neurons_num, action_num * action_size], initializer=tf.contrib.layers.xavier_initializer())
                self.po_b3 = tf.get_variable("policy_b3", [action_num * action_size], initializer=tf.constant_initializer(0.0))

                self.po_output1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_state, self.po_W1), self.po_b1))
                self.po_output2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(self.po_output1, self.po_W2), self.po_b2))
                self.po_output3 = (tf.nn.bias_add(tf.matmul(self.po_output2, self.po_W3), self.po_b3))

                self.po_probabilities = tf.nn.softmax(tf.reshape(self.po_output3, [-1, action_num, action_size]))
                self.po_max_probabilities = tf.reduce_max(self.po_probabilities, axis = -1)
                self.po_computed_actions = tf.argmax(self.po_probabilities, axis = -1)

                #self.po_matr_prev_actions = tf.reshape(self.po_prev_actions, [-1, action_num, action_size])
                self.po_eligibility = tf.log(tf.reduce_sum(tf.multiply(self.po_prev_actions, self.po_probabilities), -1))
                self.po_loss = tf.multiply(-tf.reduce_sum(self.po_eligibility, -1), self.po_return)



                self.po_learning_rate_op = tf.maximum(self.po_learning_rate_minimum,
                                                   tf.train.exponential_decay(
                                                       self.po_lrate,
                                                       self.global_step,
                                                       self.po_decay_step,
                                                       0.96,
                                                       staircase=True))

                self.po_optimizer = tf.train.AdamOptimizer(self.po_learning_rate_op).minimize(self.po_loss)

            with tf.variable_scope(self.scope + "value"):
                self.v_state = tf.placeholder(tf.float32, [None, state_size], name="v_state")
                self.v_actual_return = tf.placeholder(tf.float32, [None, 1], name="v_actual_return")

                # formula to compute number of neurons in the hidden layer: Nmin = 2*sqrt(in * out)
                hid_neurons_num = 2 * int(math.sqrt(state_size * action_num * action_size)) + 2
                hid_neurons_num = 50
                #hid_neurons_num = 32

                self.v_W1 = tf.get_variable("value_V1", [state_size, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.v_b1 = tf.get_variable("value_b1", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                self.v_W2 = tf.get_variable("value_W2", [hid_neurons_num, hid_neurons_num], initializer=tf.contrib.layers.xavier_initializer())
                self.v_b2 = tf.get_variable("value_b2", [hid_neurons_num], initializer=tf.constant_initializer(0.0))
                self.v_W3 = tf.get_variable("value_W3", [hid_neurons_num, 1], initializer=tf.contrib.layers.xavier_initializer())
                self.v_b3 = tf.get_variable("value_b3", [1], initializer=tf.constant_initializer(0.0))

                # in the future, try to calculate return for for each action

                self.v_output1 = tf.nn.bias_add(tf.matmul(self.v_state, self.v_W1), self.v_b1)
                self.v_output2 = tf.nn.bias_add(tf.matmul(self.v_output1, self.v_W2), self.v_b2)
                self.v_output3 = tf.nn.bias_add(tf.matmul(self.v_output2, self.v_W3), self.v_b3)

                self.v_diffs = self.v_output3 - self.v_actual_return
                self.v_loss = tf.nn.l2_loss(self.v_diffs)

                self.v_learning_rate_op = tf.maximum(self.v_learning_rate_minimum,
                                                   tf.train.exponential_decay(
                                                       self.v_lrate,
                                                       self.global_step,
                                                       self.v_decay_step,
                                                       0.96,
                                                       staircase=True))

                self.v_optimizer = tf.train.AdamOptimizer(self.v_learning_rate_op).minimize(self.v_loss)

            if (self.scope != ''):
                with tf.variable_scope(self.scope + "statistics"):
                    self.episode_rewards = tf.placeholder(tf.float32, [None], name="episode_rewards")
                    print self.episode_rewards.name
                    self.variable_summaries(self.episode_rewards)
                    self.merged = tf.summary.merge_all()
#            self.constructSummary(sess)

        # def constructSummary(self, sess):
        #     self.variable_summaries(self.episode_rewards)
        #     self.merged = tf.summary.merge_all()


        def calculateAction(self, sess, state):
            state_one_hot_sequence = np.expand_dims(state, axis = 0)
            action = sess.run(self.po_computed_actions, feed_dict={self.po_state : state_one_hot_sequence})
            probability = sess.run(self.po_probabilities, feed_dict={self.po_state : state_one_hot_sequence})
            return action, probability

        def calculateReward(self, sess, state):
            state_one_hot_sequence = np.expand_dims(state, axis=0)
            reward = sess.run(self.v_output3, feed_dict={self.v_state: state_one_hot_sequence})
            return reward[0][0]

        def updatePolicy(self, sess, history, step):
            rewards = history.getRewardHistory()
            advantages = []
            update_vals = []

            for i, reward in enumerate(rewards):
                future_reward = 0
                future_transitions = len(rewards) - i
                decrease = 1
                for index2 in xrange(future_transitions):
                    future_reward += rewards[(index2) + i]
                    decrease = decrease * 0.97

                prediction = self.calculateReward(sess, history.getState(i))
                advantages.append(future_reward - prediction)
                update_vals.append(future_reward)

            print self.episode_rewards.name

            _, v_lr = sess.run([self.v_optimizer, self.v_learning_rate_op], feed_dict={self.v_state: history.getStates(),
                                                                                       self.v_actual_return: np.expand_dims(update_vals, axis=1),
                                                                                       self.global_step: step
                                                                                       })

            statistics, _, W1, W3, p_lr = sess.run([self.merged, self.po_optimizer, self.po_W1, self.po_W3, self.po_learning_rate_op], feed_dict={self.po_state: history.getStates(),
                                                                                                                                                  self.po_prev_actions: history.getActions(),
                                                                                                                                                  self.po_return: advantages,
                                                                                                                                                  self.episode_rewards: history.getRewardHistory(),
                                                                                                                                                  self.global_step: step
                                                                                                                                                  })
            # _train_writer.add_summary(statistics, step)
            # _train_writer.flush()

            print "l rates", v_lr, p_lr
            return statistics
            #print W1
            #print W3

        def variable_summaries(self, var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

            print var.name
            with tf.variable_scope(self.scope + 'summaries'):
                mean = tf.reduce_mean(var)
                print var.name
                tf.summary.scalar('mean', mean)
                with tf.variable_scope(self.scope + 'stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
                tf.summary.scalar('sum', tf.reduce_sum(var))
