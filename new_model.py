"""
distributed convolutional neural network for inmage classification
This is a general model that must be adapted according to task/ dataset
"""
import os
import copy
import numpy as np
import tensorflow as tf


def from_list_to_array(data_image_list):
    batch_x = []
    batch_y = []
    for obj in data_image_list:
        batch_x.append(obj.raw_data.flatten())
        batch_y.append(obj.label)
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)
    return batch_x, batch_y


def normalize(x):
    x_norm = copy.deepcopy(x.astype(np.float32))
    x_norm /= 255.
    return x_norm

def denormalize(x_norm):
    x = copy.deepcopy(x_norm)
    x *= 255
    x = np.uint8(x)
    return x

def invert_image(batch_x):
    return 1 - batch_x


def get_conv_out_shape(
        in_shape,
        out_channels,
        kernel_y_dim,
        kernel_x_dim,
        stride_y,
        stride_x,
        padding
):
    stride_y = max(1, stride_y)
    stride_x = max(1, stride_x)
    if padding == 'SAME':
        new_y_shape = (int(in_shape[1] / stride_y)
                       + int(in_shape[1] % stride_y != 0))
        new_x_shape = (int(in_shape[2] / stride_x)
                       + int(in_shape[2] % stride_x != 0))
    elif padding == 'VALID':
        new_y_shape = (int((in_shape[1] - kernel_y_dim + 1) / stride_y)
                       + int(in_shape[1] % stride_y != 0))
        new_x_shape = (int((in_shape[2] - kernel_x_dim + 1) / stride_x)
                       + int(in_shape[2] % stride_x != 0))

    return [in_shape, new_y_shape, new_x_shape, out_channels]


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

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
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_conv_params(
        input_shape,
        out_channels,
        kernel_y_dim,
        kernel_x_dim
):
    w = tf.get_variable(
        name='conv_weight',
        shape=[kernel_y_dim, kernel_x_dim, input_shape[-1], out_channels],
        initializer=tf.truncated_normal_initializer(
            stddev=1 / np.sqrt(kernel_y_dim * kernel_x_dim * input_shape[-1])
        )
    )
    b = tf.get_variable(
        name='conv_bias',
        shape=[out_channels,],
        initializer=tf.truncated_normal_initializer()
    )
    return w, b


def get_dense_params(
        input_shape,
        dimension,
):
    w = tf.get_variable(
        name='conv_weight',
        shape=[input_shape[-1], dimension],
        initializer=tf.truncated_normal_initializer(
            stddev=1 / np.sqrt(input_shape[-1])
        )
    )
    b = tf.get_variable(
        name='conv_bias',
        shape=[dimension,],
        initializer=tf.truncated_normal_initializer()
    )
    return w, b


def conv_inference(
        in_tensor,
        weight,
        bias,
        y_stride,
        x_stride,
        padding='SAME',
        activation=tf.nn.relu
):
    intermediate = tf.nn.conv2d(
        in_tensor,
        weight,
        strides=[1, y_stride, x_stride, 1],
        padding=padding
    )
    return activation(tf.nn.bias_add(intermediate, bias))


def pool_inference(
        in_tensor,
        kernel_y_size,
        kernel_x_size,
        y_stride,
        x_stride,
        padding='SAME',
        keep_dropout=1.
):
    intermediate = tf.nn.max_pool(
        in_tensor,
        ksize=[1, kernel_y_size, kernel_x_size, 1],
        strides=[1, y_stride, x_stride, 1],
        padding=padding
    )
    return tf.nn.dropout(intermediate, keep_dropout)


def dense_inference(
        in_tensor,
        weight,
        bias,
        keep_dropout=1.,
        activation=tf.sigmoid
):
    intermediate = tf.matmul(in_tensor, weight)
    intermediate = tf.nn.bias_add(intermediate, bias)
    return activation(tf.nn.dropout(intermediate, keep_dropout))


class HWCharacterClassifier(object):
    def __init__(
            self,
            device_list,
            height,
            width,
            channels,
            output_size,
            conv_out_channels,
            conv_y_ksizes,
            conv_x_ksizes,
            conv_y_strides,
            conv_x_strides,
            conv_paddings,
            conv_activations,
            pool_y_ksizes,
            pool_x_ksizes,
            pool_y_strides,
            pool_x_strides,
            pool_paddings,
            pool_dropouts,
            dense_dimensions,
            dense_dropouts,
            dense_activations,
            folder,
            normalization_function,
            denormalization_function
    ):
        self.device_list = device_list
        self.height = height
        self.width = width
        self.channels = channels
        self.output_size = output_size
        self.conv_out_channels = conv_out_channels
        self.conv_y_ksizes = conv_y_ksizes
        self.conv_x_ksizes = conv_x_ksizes
        self.conv_y_strides = conv_y_strides
        self.conv_x_strides = conv_x_strides
        self.conv_paddings = conv_paddings
        self.conv_activations = conv_activations
        self.pool_y_ksizes = pool_y_ksizes
        self.pool_x_ksizes = pool_x_ksizes
        self.pool_y_strides = pool_y_strides
        self.pool_x_strides = pool_x_strides
        self.pool_paddings = pool_paddings
        # self.pool_dropouts = pool_dropouts
        self.dense_dimensions = dense_dimensions
        # self.dense_dropouts = dense_dropouts
        self.dense_activations = dense_activations
        self.normalize = normalization_function
        self.denormalize = denormalization_function

        self.folder = folder

        self.input_placeholder = None
        self.label_placeholder = None
        self.params = None
        self.conv_dropout_ph = None
        self.dense_dropout_ph = None

        self.sess = None
        self.global_step = 0

        self.set_graph()

    def set_graph(self):
        data_placeholders_device = '/cpu:0'
        parameters_device = '/cpu:0'
        split_data_device = '/cpu:0'
        optimizer_device = '/cpu:0'
        averaging_device = '/cpu:0'
        inference_device = self.device_list[0]

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
            input_split = tf.split(
                self.input_placeholder,
                len(self.device_list),
                axis=0,
                name='split_input'
            )
            label_split = tf.split(
                self.label_placeholder,
                len(self.device_list),
                axis=0,
                name='split_label'
            )

        with tf.device(optimizer_device):
            self.lr = tf.placeholder(tf.float32, shape=(), name='lr_placeholder')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.grad_and_vars = []
        costs = []
        for i, d in enumerate(self.device_list):
            with tf.name_scope('tower_{}'.format(i)), tf.device(d):
                this_cost = self.get_cost(input_split[i], label_split[i])
                costs.append(this_cost)
                # self.grad_and_vars.append(self.optimizer.compute_gradients(this_cost, var_list=self.list_var))
                self.grad_and_vars.append(self.optimizer.compute_gradients(this_cost))

        with tf.name_scope('averaging'), tf.device(averaging_device):
            self.grads = average_gradients(self.grad_and_vars)
            self.apply_gradients = self.optimizer.apply_gradients(self.grads)
            self.cost = tf.add_n(costs) / len(self.device_list)

        with tf.name_scope('plain_inference'), tf.device(inference_device):
            self.model_output = self.model_inference(self.input_placeholder)

            self.model_prediction = tf.arg_max(tf.nn.softmax(self.model_output), 1)
            self.correct_prediction = tf.equal(tf.arg_max(self.model_output, 1),
                                               tf.arg_max(self.label_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def create_parameters(self):
        assert len(self.conv_out_channels) == len(self.conv_y_ksizes)
        assert len(self.conv_out_channels) == len(self.conv_x_ksizes)
        assert len(self.conv_out_channels) == len(self.conv_y_strides)
        assert len(self.conv_out_channels) == len(self.conv_x_strides)
        assert len(self.conv_out_channels) == len(self.conv_paddings)
        assert len(self.conv_out_channels) == len(self.pool_y_ksizes)
        assert len(self.conv_out_channels) == len(self.pool_x_ksizes)
        assert len(self.conv_out_channels) == len(self.pool_y_strides)
        assert len(self.conv_out_channels) == len(self.pool_x_strides)
        assert len(self.conv_out_channels) == len(self.pool_paddings)
#        assert len(self.conv_out_channels) == len(self.pool_dropouts)

#        assert len(self.dense_dropouts) == len(self.dense_dimensions)

        last_shape = [-1, self.height, self.width, self.channels]
        self.params = []
        self.conv_dropout_ph = []
        self.dense_dropout_ph = []
        for e in range(len(self.conv_out_channels)):
            out_channel = self.conv_out_channels[e]
            c_y_ksize = self.conv_y_ksizes[e]
            c_x_ksize = self.conv_x_ksizes[e]
            c_y_stride = self.conv_y_strides[e]
            c_x_stride = self.conv_x_strides[e]
            c_padding = self.conv_paddings[e]
            p_y_ksize = self.pool_y_ksizes[e]
            p_x_ksize = self.pool_x_ksizes[e]
            p_y_stride = self.pool_y_strides[e]
            p_x_stride = self.pool_x_strides[e]
            p_padding = self.pool_paddings[e]

            conv_dropout_ph = tf.placeholder_with_default(input=tf.constant(1., dtype=tf.float32, shape=()),
                                                          shape=(),
                                                          name='conv_{}_dropout_ph'.format(e))
            self.conv_dropout_ph.append(conv_dropout_ph)

            with tf.variable_scope('conv_{}'.format(e)):
                weight, bias = get_conv_params(
                    last_shape,
                    out_channel,
                    c_y_ksize,
                    c_x_ksize,
                )
            self.params.append(weight)
            self.params.append(bias)
            last_shape = get_conv_out_shape(
                in_shape=last_shape,
                out_channels=out_channel,
                kernel_y_dim=c_y_ksize,
                kernel_x_dim=c_x_ksize,
                stride_y=c_y_stride,
                stride_x=c_x_stride,
                padding=c_padding
            )
            last_shape = get_conv_out_shape(
                in_shape=last_shape,
                out_channels=last_shape[-1],
                kernel_y_dim=p_y_ksize,
                kernel_x_dim=p_x_ksize,
                stride_y=p_y_stride,
                stride_x=p_x_stride,
                padding=p_padding
            )

        last_shape = [last_shape[0], np.asarray(last_shape[1:]).prod()]

        for e, dimension in enumerate(self.dense_dimensions):
            with tf.variable_scope('dense_{}'.format(e)):
                weight, bias = get_dense_params(
                    last_shape,
                    dimension
                )
            self.params.append(weight)
            self.params.append(bias)
            last_shape = [last_shape[0], dimension]
            dense_dropout_ph = tf.placeholder_with_default(input=tf.constant(1., dtype=tf.float32, shape=()),
                                                           shape=(),
                                                           name='dense_{}_dropout_ph'.format(e))
            self.dense_dropout_ph.append(dense_dropout_ph)

    def get_cost(self, input_tensor, target_tensor):
        prediction = self.model_inference(input_tensor)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_tensor, logits=prediction))
        return cross_entropy

    def model_inference(self, input_tensor):
        last_shape = [-1, self.height, self.width, self.channels]
        last_tensor = tf.reshape(input_tensor, [-1, self.height, self.width, self.channels])
        for e in range(len(self.conv_out_channels)):
            out_channel = self.conv_out_channels[e]
            c_y_ksize = self.conv_y_ksizes[e]
            c_x_ksize = self.conv_x_ksizes[e]
            c_y_stride = self.conv_y_strides[e]
            c_x_stride = self.conv_x_strides[e]
            c_padding = self.conv_paddings[e]
            p_y_ksize = self.pool_y_ksizes[e]
            p_x_ksize = self.pool_x_ksizes[e]
            p_y_stride = self.pool_y_strides[e]
            p_x_stride = self.pool_x_strides[e]
            p_padding = self.pool_paddings[e]
            p_dropout = self.conv_dropout_ph[e]
            activation = self.conv_activations[e]

            weight = self.params[2 * e]
            bias = self.params[2 * e + 1]

            with tf.name_scope('conv_{}_inference'.format(e)):
                last_tensor = conv_inference(
                    last_tensor,
                    weight,
                    bias,
                    y_stride=c_y_stride,
                    x_stride=c_x_stride,
                    padding=c_padding,
                    activation=activation
                )

            if p_y_ksize > 0:
                with tf.name_scope('pool_{}_inference'.format(e)):
                    last_tensor = pool_inference(
                        last_tensor,
                        kernel_y_size=p_y_ksize,
                        kernel_x_size=p_x_ksize,
                        y_stride=p_y_stride,
                        x_stride=p_x_stride,
                        padding=p_padding,
                        keep_dropout=p_dropout,
                    )

        last_tensor = tf.reshape(
            last_tensor,
            [-1, np.asarray(last_tensor.get_shape().as_list()[1:]).prod()]
            )

        for e in range(len(self.dense_dimensions)):
            dimension = self.dense_dimensions[e]
            dropout = self.dense_dropout_ph[e]
            activation = self.dense_activations[e]
            weight = self.params[2 * len(self.conv_out_channels) + 2 * e]
            bias = self.params[2 * len(self.conv_out_channels) + 2 * e + 1]

            with tf.name_scope('dense_{}_inference'.format(e)):
                last_tensor = dense_inference(last_tensor, weight, bias, dropout, activation)

        # last_tensor = tf.nn.softmax(last_tensor)

        return last_tensor

    def open_session_2(self, max_nr_of_savings_to_keep=20, load_params=True):
        if self.sess is None:
            self.saver = tf.train.Saver(max_to_keep=max_nr_of_savings_to_keep)
            init = tf.global_variables_initializer()
            config = tf.ConfigProto(log_device_placement=False)
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
                print 'Parameters restored from {}'.format(save_path)

            else:
                self.sess.run(init)
                print 'Parameters randomly initialized'

        return self.sess

    def close_session(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def predict(self, data):
        sess = self.open_session()
        data_norm, _ = self.normalize(data)
        res = sess.run(
            self.model_output,
            feed_dict={self.input_placeholder: data_norm}
        )
        _, res_denorm = self.denormalize(y_norm=res)
        return res_denorm

    def predict_2(self, data):
        sess = self.open_session_2()
        data_norm, _ = self.normalize(data)
        res = sess.run(
            self.model_output,
            feed_dict={self.input_placeholder: data_norm}
        )
        _, res_denorm = self.denormalize(y_norm=res)
        return res_denorm


    def train2(self,
               train_dataset,
               test_dataset,
               n_iterations,
               train_batch_size,
               test_batch_size,
               learning_rate,
               frequency_print,
               frequency_save,
               learning_rate_scale_factor=1.0,
               learning_rate_update_factor=1000,
               conv_dropout_vals=None,
               dense_dropout_vals=None,
               max_nr_of_savings_to_keep=20,
               load_params=False,
               save_params=True
               ):

        self.open_session_2(max_nr_of_savings_to_keep=max_nr_of_savings_to_keep,
                            load_params=load_params)

        MIN_LEARNING_RATE = 10 ** (-6)

        for step in range(n_iterations):
            if step % learning_rate_update_factor == 0 and step > 0 and learning_rate > MIN_LEARNING_RATE:
                learning_rate *= learning_rate_scale_factor
                print '\n --- Learning rate set to ', learning_rate, '\n'

            x_batch, y_batch = train_dataset.next_batch(train_batch_size)

            x_batch_norm = self.normalize(x_batch)
            x_batch_norm = invert_image(x_batch_norm)

            fdict = {self.input_placeholder: x_batch_norm.reshape([-1, self.height * self.width * self.channels]),
                     self.label_placeholder: y_batch,
                     self.lr: learning_rate
                     }
            if conv_dropout_vals is not None:
                for ph, val in zip(self.conv_dropout_ph, conv_dropout_vals):
                    fdict[ph] = val
            if dense_dropout_vals is not None:
                for ph, val in zip(self.dense_dropout_ph, dense_dropout_vals):
                    fdict[ph] = val

            _, train_cost = self.sess.run([self.apply_gradients, self.cost],
                                          feed_dict=fdict
                                          )
            if step % frequency_print == 0:
                x_b, y_b = test_dataset.next_batch(test_batch_size)
                x_b_norm = self.normalize(x_b)
                x_b_norm = invert_image(x_b_norm)

                train_accuracy = self.sess.run(self.accuracy, feed_dict=fdict)
                test_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.input_placeholder: x_b_norm.reshape([-1, self.height * self.width * self.channels]),
                    self.label_placeholder: y_b,
                    self.lr: learning_rate
                    })

                print('{:17}iter {:>4}   Train Accuracy:  {:>5.2f}%  Test Accuracy:  {:>5.2f}%'.format('',
                                                                                                       step,
                                                                                                       train_accuracy * 100,
                                                                                                       test_accuracy * 100))
            if save_params and step % frequency_save == 0 and step != 0:
                print 'Saving params...'
                dirname = os.path.join(self.folder, 'checkpoints')
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                path = self.saver.save(self.sess, os.path.join(dirname, 'epoch'),
                                       global_step=self.global_step)
                print('Data saved to {}'.format(path))

            self.global_step += 1
        self.close_session()







