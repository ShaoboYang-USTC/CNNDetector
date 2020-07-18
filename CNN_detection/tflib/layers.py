import tensorflow as tf


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            std_summ = tf.summary.scalar('stddev', stddev)
            max_summ = tf.summary.scalar('max', tf.reduce_max(var))
            min_summ = tf.summary.scalar('min', tf.reduce_min(var))
            his_summ = tf.summary.histogram('histogram', var)

    tf.add_to_collection('train_summary', std_summ)
    tf.add_to_collection('train_summary', max_summ)
    tf.add_to_collection('train_summary', min_summ)
    tf.add_to_collection('train_summary', his_summ)


def conv(input,
         filter,
         strides,
         padding,
         acti_func=tf.nn.relu,
         wd=None,
         bias=None,
         name=None):
    with tf.variable_scope(name) as scope:
        # kernel = tf.get_variable('weight',
        #                          shape=filter,
        #                          dtype=tf.float32,
        #                          initializer=tf.truncated_normal_initializer(stddev=0.1))
        kernel = tf.Variable(initial_value=tf.truncated_normal(filter, stddev=0.1), name='weight')
        variable_summaries(kernel, 'weight')

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(kernel), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        if bias is not None:
            # bias = tf.get_variable('bias',
            #                        filter[-1],
            #                        dtype=tf.float32,
            #                        initializer=tf.constant_initializer(bias))
            bias = tf.Variable(tf.constant(bias, shape=[filter[-1]]), name='bias')
            variable_summaries(bias, 'bias')

        convolution = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
        act = acti_func(convolution + bias, name='activation')
        his_summ = tf.summary.histogram('activations', act)
        tf.add_to_collection('train_summary', his_summ)
        return act


def pool(input,
         ksize=[1, 1, 3, 1],
         strides=[1, 1, 3, 1],
         padding='SAME',
         pool_func=tf.nn.max_pool,
         name=None):
    with tf.variable_scope(name) as scope:
        return pool_func(input, ksize=ksize, strides=strides, padding=padding, name=name)


def unfold(input, name=None):
    with tf.variable_scope(name) as scope:
        input = tf.reduce_mean(input,1)
        num_batch, width, num_channels = input.get_shape()
        #num_batch, height, width, num_channels = input.get_shape()
        output = tf.reshape(input, [-1, num_channels])
        return output


def fc(input,
       output_dim,
       input_dim=None,
       acti_func=tf.nn.relu,
       wd=None,
       name=None):
    with tf.variable_scope(name) as scope:
        # input_dim = tf.shape(input)[1]
        if input_dim is None:
            num_batch, input_dim = input.get_shape()
            input_dim = input_dim.value
        weights = tf.get_variable('weight',
                                  shape=[input_dim, output_dim],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        # weights = tf.Variable(tf.truncated_normal(shape=[input_dim.value, output_dim], stddev=0.1), name='weight')
        variable_summaries(weights, 'weight')
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        bias = tf.get_variable('bias',
                               output_dim,
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        # bias = tf.Variable(tf.constant(0.0, shape=[output_dim]), name='bias')
        variable_summaries(bias, 'bias')
        output = tf.matmul(input, weights) + bias
        output = acti_func(output)
        his_summ = tf.summary.histogram('activations', output)
        tf.add_to_collection('train_summary', his_summ)
        return output

