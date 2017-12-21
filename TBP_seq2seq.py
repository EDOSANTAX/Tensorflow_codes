
import numpy as np
import tensorflow as tf
from Datasets_sequences import time_as_space_seq
import matplotlib.pyplot as plt
import os

"""
This is a rough experimental implmentation of the truncated backpropagation through time for a LSTM

"""


def reshape_data(data,
                 long_memory,
                 short_memory,
                 n_inputs,
                 stride
                 ):
    data_3d = np.reshape(data, (-1, long_memory, n_inputs))
    data_list = []
    for i in range(0, long_memory, stride):
        data_block = data_3d[:, i:i + short_memory, :]
        data_list.append(data_block)

    return data_list


def my_LSTM(stack):

    lstm_cells = []
    for size in stack:
        cell = tf.nn.rnn_cell.BasicLSTMCell(size,
                                            forget_bias=1.0,
                                            state_is_tuple=True)

        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.)

        # lstm_cells.append(tf.nn.rnn_cell.BasicLSTMCell(size,
        #                                                forget_bias=1.0,
        #                                                state_is_tuple=True))

        lstm_cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(lstm_cells)


def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def lstm_within_memory(
                      input_3d_tensor,
                      short_memory,
                      n_inputs,
                      lstm_network_model,
                      initial_state=None,
                      ):

    # Permuting batch_size and memory
    observations_2 = tf.transpose(input_3d_tensor, [1, 0, 2])
    # Reshape to (memory*batch_size, n_inputs)
    observations_3 = tf.reshape(observations_2, [-1, n_inputs])
    # Splitting in list
    observations_4 = tf.split(num_or_size_splits=short_memory, value=observations_3, axis=0)

    outputs = []
    states = []
    with tf.variable_scope('my_LSTM'):
        for i in range(short_memory):

            if i == 0:
                lstm_outputs, lstm_states = lstm_network_model.__call__(observations_4[i], initial_state)
                outputs.append(lstm_outputs)
                states.append(lstm_states)
            else:
                tf.get_variable_scope().reuse_variables()
                lstm_outputs, lstm_states = lstm_network_model.__call__(observations_4[i], lstm_states)
                outputs.append(lstm_outputs)
                states.append(lstm_states)

    return outputs, states


def adapt_2_prediction(rnn_outputs,
                       prediction_size
                       ):

    in_dim = rnn_outputs[0].get_shape()[1]

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [in_dim, prediction_size])
        b = tf.get_variable('b', [prediction_size], initializer=tf.constant_initializer(0.0))
    adapted_outputs = [tf.expand_dims(tf.matmul(rnn_output, W) + b, axis=1) for rnn_output in rnn_outputs]

    return adapted_outputs


def model_inference(
                    lstm_outs,
                    lstm_states,
                    target_size
                    ):

    out_tensor_list = adapt_2_prediction(rnn_outputs=lstm_outs,
                                         prediction_size=target_size)

    return tf.concat(out_tensor_list, axis=1), lstm_states


def get_cost(output_tensor, target_tensor):
    return tf.reduce_mean(tf.squared_difference(output_tensor, target_tensor))


def run_one_training_iteration(
                               data_x_list,
                               data_y_list,
                               final_state=None,
                               display_cost=False
                               ):

    outputs = []
    costs = []
    grads_and_vars = []

    for i in range(len(data_x_list)):
        batch_x = data_x_list[i]
        batch_y = data_y_list[i]

        if final_state is None:
            if i == 0:
                states = sess.run(stt, feed_dict={observations: batch_x})
                final_state = states[-1]
                out_within_mem = sess.run(out, feed_dict={observations: batch_x})
                outputs.append(out_within_mem)
                costs.append(sess.run(this_cost, feed_dict={observations: batch_x, targets: batch_y}))
                grads_and_vars.append(sess.run(g_v_couples, feed_dict={observations: batch_x, targets: batch_y}))
            else:
                feed_dict = {observations: batch_x, targets: batch_y}
                for j, (c, h) in enumerate(final_state):
                    feed_dict[states_0[j].c] = final_state[j].c
                    feed_dict[states_0[j].h] = final_state[j].h

                states = sess.run(stt, feed_dict=feed_dict)
                out_within_mem = sess.run(out, feed_dict=feed_dict)
                outputs.append(out_within_mem)
                costs.append(sess.run(this_cost, feed_dict=feed_dict))
                grads_and_vars.append(sess.run(g_v_couples, feed_dict=feed_dict))
                final_state = states[-1]

        else:
            feed_dict = {observations: batch_x, targets: batch_y}
            for j, (c, h) in enumerate(final_state):
                feed_dict[states_0[j].c] = final_state[j].c
                feed_dict[states_0[j].h] = final_state[j].h

            states = sess.run(stt, feed_dict=feed_dict)
            out_within_mem = sess.run(out, feed_dict=feed_dict)
            outputs.append(out_within_mem)
            costs.append(sess.run(this_cost, feed_dict=feed_dict))
            grads_and_vars.append(sess.run(g_v_couples, feed_dict=feed_dict))
            final_state = states[-1]

    averaged_grads = average_gradients(grads_and_vars)
    #print len(averaged_grads)
    feed_dict_2 = {}

    for i in range(len(averaged_grads)):
        # print sess.run(averaged_grads[i][0])
        feed_dict_2[g_v_couples[i][0]] = sess.run(averaged_grads[i][0])
        # feed_dict_2[g_v_couples[i][1]] = averaged_grads[i][1]

    sess.run(apply_grads, feed_dict=feed_dict_2)

    if display_cost:
        print costs[-1]

    return costs, final_state


def next_bath(data_x, data_y, b_size):

    shuffled_indices = np.arange(len(data_x))
    np.random.shuffle(shuffled_indices)
   
    selected_indices = np.random.choice(shuffled_indices, size=b_size, replace=False)

    return data_x[selected_indices], data_y[selected_indices]



if __name__ == '__main__':

    train_x, train_y, test_x, test_y = time_as_space_seq(train_size=1000, memory=50)


    train_x = np.reshape(train_x, (1000, -1))
    train_y = np.reshape(train_y, (1000, -1))

    test_x = np.reshape(test_x, (test_x.shape[0], -1))
    test_y = np.reshape(test_y, (test_y.shape[0], -1))

    #------------------------------- PARAMETERS -------------------------------------------------------------------------
    LONG_MEMORY = 50
    SHORT_MEMORY = 10
    STRIDE = 10
    N_INPUTS = 3
    STACK = [7, 5]
    BATCH_SIZE = 100
    #--------------------------------------------------------------------------------------------------------------------

    train_x_list = reshape_data(data=train_x,
                                long_memory=LONG_MEMORY,
                                short_memory=SHORT_MEMORY,
                                n_inputs=N_INPUTS,
                                stride=STRIDE)

    train_y_list = reshape_data(data=train_y,
                                long_memory=LONG_MEMORY,
                                short_memory=SHORT_MEMORY,
                                n_inputs=N_INPUTS,
                                stride=STRIDE)

    ## BUILDING THE GRAPH

    observations = tf.placeholder(tf.float32,
                                  [None, SHORT_MEMORY, N_INPUTS],
                                  name='observations')

    targets = tf.placeholder(tf.float32,
                                  [None, SHORT_MEMORY, N_INPUTS],
                                  name='targets')

    lstm_network = my_LSTM(stack=STACK)

    states_0 = lstm_network.zero_state(BATCH_SIZE, tf.float32)
    print type(states_0[0])

    lstm_outs, lstm_states = lstm_within_memory(input_3d_tensor=observations,
                                                short_memory=SHORT_MEMORY,
                                                n_inputs=N_INPUTS,
                                                lstm_network_model=lstm_network,
                                                initial_state=states_0)

    out, stt = model_inference(lstm_outs=lstm_outs,
                           lstm_states=lstm_states,
                           target_size=N_INPUTS)


    this_cost = get_cost(output_tensor=out,
                         target_tensor=targets)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    g_v_couples = optimizer.compute_gradients(this_cost)
    print '****************************************'
    print g_v_couples[0]
    apply_grads = optimizer.apply_gradients(g_v_couples)

    # print('\n---------------------------------------\n')
    # for v in tf.global_variables():
    #     print v
    # print('\n---------------------------------------\n')

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    ## RUNNING THE GRAPH

    ##using function-------------------------------------------------

    for e in range(1000):

        if e%10 == 0:
            print_cost = True
            print '***************************+'
            print 'iterationn ', e

            t_x_test, t_y_test = next_bath(test_x, test_y, BATCH_SIZE)
            t_x_list_test = reshape_data(data=t_x_test,
                                    long_memory=LONG_MEMORY,
                                    short_memory=SHORT_MEMORY,
                                    n_inputs=N_INPUTS,
                                    stride=STRIDE)
            t_y_list_test = reshape_data(data=t_y_test,
                                    long_memory=LONG_MEMORY,
                                    short_memory=SHORT_MEMORY,
                                    n_inputs=N_INPUTS,
                                    stride=STRIDE)
            print 'test test test'
            c_l_test, f_s_test = run_one_training_iteration(t_x_list_test, t_y_list_test, display_cost=True)


        else:
            print_cost = False
        # t_x = train_x[e*BATCH_SIZE:e*BATCH_SIZE+BATCH_SIZE,:]
        # t_y = train_y[e*BATCH_SIZE:e*BATCH_SIZE+BATCH_SIZE,:]

        t_x, t_y = next_bath(train_x,train_y,BATCH_SIZE)
        t_x_list = reshape_data(data=t_x,
                                long_memory=LONG_MEMORY,
                                short_memory=SHORT_MEMORY,
                                n_inputs=N_INPUTS,
                                stride=STRIDE)

        t_y_list = reshape_data(data=t_y,
                                long_memory=LONG_MEMORY,
                                short_memory=SHORT_MEMORY,
                                n_inputs=N_INPUTS,
                                stride=STRIDE)


        if e == 0:
            cost_list, f_s = run_one_training_iteration(t_x_list, t_y_list, display_cost=print_cost)
        else:

            # cost_list, f_s = run_one_training_iteration(t_x_list, t_y_list, f_s)
            cost_list, f_s = run_one_training_iteration(t_x_list, t_y_list, display_cost=print_cost)

    print cost_list
    #--------------------------------------------------------------------------------

