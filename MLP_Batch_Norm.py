"""
Multilayer Perceptron trained on classification task (MNIST handwritten digits)
adopting the Batch Normalization technique
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_dense_params(
                     input_shape,
                     dimension,
                     BN_layer=True
                     ):
    ## Creates weights and biases for a dense layer
    w = tf.get_variable(
        name='conv_weight',
        shape=[input_shape[-1], dimension],
        initializer=tf.truncated_normal_initializer(
            stddev=1 / np.sqrt(input_shape[-1])
        )
    )
    if not BN_layer:
        b = tf.get_variable(
            name='conv_bias',
            shape=[dimension,],
            initializer=tf.truncated_normal_initializer()
        )
        return w, b
    else:
        return w


def get_BN_parameters(inputs_size):

    ## Creates the parameters necessary for the Batch Normalization

    scale = tf.get_variable(name='scale',
                            shape=[inputs_size],
                            initializer=tf.zeros_initializer
                            )
    beta = tf.get_variable(name='beta',
                            shape=[inputs_size],
                            initializer=tf.zeros_initializer
                           )
    pop_mean = tf.get_variable(name='pop_mean',
                            shape=[inputs_size],
                            initializer=tf.zeros_initializer,
                            trainable=False
                            )
    pop_var = tf.get_variable(name='pop_var',
                            shape=[inputs_size],
                            initializer=tf.zeros_initializer,
                            trainable=False
                            )
    return scale, beta, pop_mean, pop_var


def my_batch_norm(in_tensor,
                  mean,
                  var,
                  beta,
                  scale,
                  epsilon=1e-3
                  ):
    z_hat = (in_tensor - mean) / tf.sqrt(var + epsilon)
    BN_out = scale * z_hat + beta
    return BN_out


def batch_norm_wrapper(
                       inputs,
                       scale,
                       beta,
                       pop_mean,
                       pop_var,
                       is_training,
                       decay = 0.999,
                       epsilon=1e-3
                       ):
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return my_batch_norm(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return my_batch_norm(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def dense_inference(
                    in_tensor,
                    weight,
                    bias=None,
                    keep_dropout=1.,
                    activation=tf.sigmoid,
                    apply_BN=True,
                    is_training=True,
                    scale=None,
                    beta=None,
                    pop_mean=None,
                    pop_var=None
                    ):
    ## Computes the output of a dense layer
    if not apply_BN:
        intermediate = tf.matmul(in_tensor, weight)
        intermediate = tf.nn.bias_add(intermediate, bias)
        return activation(tf.nn.dropout(intermediate, keep_dropout))
    else:
        intermediate = tf.matmul(in_tensor, weight)
        intermediate_bn = batch_norm_wrapper(inputs=intermediate,
                                             scale=scale,
                                             beta=beta,
                                             pop_mean=pop_mean,
                                             pop_var=pop_var,
                                             is_training=is_training)
        return activation(intermediate_bn)


class MLP_BN(object):
    def __init__(
                 self,
                 device_list,
                 height,
                 width,
                 channels,
                 output_size,
                 dense_dimensions,
                 dense_dropouts,
                 dense_activations,
                 dense_BN_layer,
                 folder
                 ):
        self.device_list = device_list
        self.height = height
        self.width = width
        self.channels = channels
        self.output_size = output_size
        self.dense_dimensions = dense_dimensions
        self.dense_dropouts = dense_dropouts
        self.dense_activations = dense_activations
        self.dense_BN_layer = dense_BN_layer
        self.folder = folder
        self.input_placeholder = None
        self.label_placeholder = None
        self.params_w = None
        self.params_b = None
        self.sess = None
        self.global_step = 0

    def set_graph(self, is_training=True):

        data_placeholders_device = '/cpu:0'
        parameters_device = '/cpu:0'
        split_data_device = '/cpu:0'
        optimizer_device = '/cpu:0'
        averaging_device = '/cpu:0'
        inference_device = self.device_list[0]
        ##note that as we compute the average layer activation within a batch
        ##it would be incorrect to split the computation over multiple gpus

        with tf.name_scope('data_placeholder'), tf.device(data_placeholders_device):
            self.input_placeholder = tf.placeholder(
                                        dtype=tf.float32,
                                        shape=[None, self.height * self.width * self.channels],
                                        name='input_placeholder'
                                        )
            self.label_placeholder = tf.placeholder(
                                        dtype=tf.float32,
                                        shape=[None, self.output_size],
                                        name='label_placeholder'
                                        )
        with tf.variable_scope('model_parameters'), tf.device(parameters_device):
            self.create_parameters()

        with tf.name_scope('split_data_tensor'), tf.device(split_data_device):
            input_split = tf.split(value=self.input_placeholder,
                                   num_or_size_splits=len(self.device_list),
                                   axis=0,
                                   name='input_split')

            label_split = tf.split(value=self.label_placeholder,
                                   num_or_size_splits=len(self.device_list),
                                   axis=0,
                                   name='input_split')

        with tf.device(optimizer_device):
            self.lr = tf.placeholder(tf.float32, shape=(), name='lr_placeholder')
            optimizer = tf.train.AdamOptimizer(self.lr)

        self.grads_and_vars = []
        costs = []
        for i, d in enumerate(self.device_list):
            with tf.name_scope('tower_{}'.format(i)), tf.device(d):
                this_cost = self.get_cost(input_tensor=input_split[i],
                                          target_tensor=label_split[i])
                costs.append(this_cost)
                self.grads_and_vars.append(optimizer.compute_gradients(loss=this_cost))

        with tf.name_scope('averaging'), tf.device(averaging_device):
            self.grads = average_gradients(self.grads_and_vars)
            self.train_step = optimizer.apply_gradients(grads_and_vars=self.grads)

        with tf.name_scope('plain_inference'), tf.device(inference_device):
            self.prediction = self.model_inference(input_tensor=self.input_placeholder,
                                                   is_training=is_training
                                                   )
            self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                               tf.argmax(self.label_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def create_parameters(self):

        self.params_w = []
        self.params_b = []
        self.scales = []
        self.betas = []
        self.pop_means = []
        self.pop_vars = []

        assert len(self.dense_dropouts) == len(self.dense_dimensions)
        last_shape = [-1, self.height*self.width*self.channels]

        for e, dimension in enumerate(self.dense_dimensions):
            if self.dense_BN_layer[e] == True:
                with tf.variable_scope('dense_{}'.format(e)):
                    weight= get_dense_params(input_shape=last_shape,
                                             dimension=dimension,
                                             BN_layer=True
                                             )
                with tf.variable_scope('BN_params_{}'.format(e)):
                    scale, beta, pop_mean, pop_var = get_BN_parameters(inputs_size=dimension)

                self.params_w.append(weight)
                self.params_b.append('_')
                self.scales.append(scale)
                self.betas.append(beta)
                self.pop_means.append(pop_mean)
                self.pop_vars.append(pop_var)

            else:
                with tf.variable_scope('dense_{}'.format(e)):
                    weight, bias= get_dense_params(input_shape=last_shape,
                                             dimension=dimension,
                                             BN_layer=False
                                             )
                self.params_w.append(weight)
                self.params_b.append(bias)
                self.scales.append('_')
                self.betas.append('_')
                self.pop_means.append('_')
                self.pop_vars.append('_')

            last_shape = [last_shape[0], dimension]

    def model_inference(self,
                        input_tensor,
                        is_training
                        ):
        last_tensor = input_tensor
        for e in range(len(self.dense_dimensions)):
            if self.dense_BN_layer[e] == True:
                weight = self.params_w[e]
                activation = self.dense_activations[e]
                dropout = self.dense_dropouts[e]
                scale = self.scales[e]
                beta = self.betas[e]
                pop_mean = self.pop_means[e]
                pop_var = self.pop_vars[e]
                with tf.name_scope('dense_{}_inference'.format(e)):
                    last_tensor = dense_inference(in_tensor=last_tensor,
                                                  weight=weight,
                                                  bias=None,
                                                  keep_dropout=dropout,
                                                  activation=activation,
                                                  apply_BN=True,
                                                  is_training=is_training,
                                                  scale=scale,
                                                  beta=beta,
                                                  pop_mean=pop_mean,
                                                  pop_var=pop_var
                                                  )
            else:
                weight = self.params_w[e]
                bias = self.params_b[-1]
                activation = self.dense_activations[e]
                dropout = self.dense_dropouts[e]
                with tf.name_scope('dense_{}_inference'.format(e)):
                    last_tensor = dense_inference(in_tensor=last_tensor,
                                                  weight=weight,
                                                  bias=bias,
                                                  keep_dropout=dropout,
                                                  activation=activation,
                                                  apply_BN=False,
                                                  is_training=is_training)
        return last_tensor

    def get_cost(self,
                 input_tensor,
                 target_tensor,
                 ):
        prediction = self.model_inference(input_tensor=input_tensor,
                                          is_training=True
                                          )
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_tensor,
                                                                               logits=prediction))
        return cross_entropy

    def open_session(self, load_params=True):
        if self.sess is None:
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto(log_device_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            if not os.path.exists(self.folder):
                os.makedirs(self.folder)

            if load_params and os.path.exists(os.path.join(self.folder, 'checkpoints')):

                files = sorted(os.listdir(os.path.join(self.folder, 'checkpoints')))
                files = [f.split('.')[0] for f in files if f.endswith('index')]
                ixs = sorted([int(name.split('-')[-1]) for name in files])
                save_path = os.path.join(self.folder, 'checkpoints', 'epoch-{}'.format(ixs[-1]))
                self.global_step = ixs[-1]
                self.saver.restore(self.sess, save_path)
                print 'Parmaeters restored from {}'.format(save_path)

            else:
                self.sess.run(init)
                print 'Parameters randomly initialized'

        return self.sess

    def close_session(self):
        if self.sess is not None:
            self.sess.close
            self.sess = None

    def train(
            self,
            ds,
            n_epochs,
            batch_size,
            learning_rate,
            frequency_print,
            frequency_save
            ):
        sess = self.open_session()

        for step in range(n_epochs):
            x_batch, y_batch = ds.train.next_batch(batch_size)

            sess.run(self.train_step, feed_dict={self.input_placeholder:x_batch,
                                                 self.label_placeholder:y_batch,
                                                 self.lr:learning_rate}
                                                 )
            if step % frequency_print == 0:
                xs, ys = ds.test.next_batch(batch_size)
                train_accuracy = sess.run(self.accuracy,
                                          feed_dict={self.input_placeholder: x_batch,
                                                     self.label_placeholder: y_batch})

                test_accuracy = sess.run(self.accuracy,
                                         feed_dict={self.input_placeholder: xs,
                                                    self.label_placeholder: ys})

                print('Epoch # {}   Train Accuracy:  {}  Test Accuracy:  {}'.format(step, train_accuracy, test_accuracy))

            if step % frequency_save == 0:
                path = self.saver.save(sess, os.path.join(self.folder, 'checkpoints', 'epoch'),
                                       global_step=self.global_step)
                print('Data saved to {}'.format(path))
            self.global_step += 1

        self.close_session()

