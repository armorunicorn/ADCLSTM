# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pickle


class double_aclstm_clf(object):
    def __init__(self, config):
        self.max_length = config.max_length
        self.char_max_length = config.char_max_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
        self.num_filters = config.num_filters
        self.hidden_size = len(self.filter_sizes) * self.num_filters
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda
        self.attn_size = config.attn_size
        self._initial_state = list()

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.input_char_x = tf.placeholder(dtype=tf.int32, shape=[None, self.char_max_length])
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
        self.char_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.Variable(self.get_embding(config.embd_file, config.embedding_size),
                                    name="embedding")
            embed = tf.nn.embedding_lookup(embedding, self.input_x)
            inputs = tf.expand_dims(embed, -1)

            char_embedding = tf.Variable(self.get_embding(config.char_embd_file, config.char_embedding_size),
                                    name="char_embedding")
            char_embed = tf.nn.embedding_lookup(char_embedding, self.input_char_x)
            char_inputs = tf.expand_dims(char_embed, -1)
        # print(inputs)
        # Input dropout
        outputs1 = self.create_clstm(inputs, self.max_length, 1)
        outputs2 = self.create_clstm(char_inputs, self.char_max_length, 2)
        outputs = tf.concat([outputs1, outputs2], 1)

        self.final_state = outputs

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size * 2, self.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            # logits
            print(softmax_w.get_shape())
            self.logits = tf.matmul(self.final_state, softmax_w) + softmax_b
            # self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1)

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32), name='correct_num')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def create_clstm(self, inputs, inputs_length, num):
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        conv_outputs = []
        max_feature_length = inputs_length - max(self.filter_sizes) + 1
        # Convolutional layer with different lengths of filters in parallel
        # No max-pooling
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('conv-%s' % filter_size):
                # [filter size, embedding size, channels, number of filters]
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable('weights_%d' % num, filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('biases_%d' % num, [self.num_filters], initializer=tf.constant_initializer(0.0))

                # Convolution
                conv = tf.nn.conv2d(inputs,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')

                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                """pooled = tf.nn.avg_pool(
                    h,
                    ksize=[1, self.max_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")"""

                # Remove channel dimension
                h_reshape = tf.squeeze(h, [2])
                # h_reshape = tf.squeeze(pooled, [2])
                # Cut the feature sequence at the end based on the maximum filter length
                h_reshape = h_reshape[:, :max_feature_length, :]

                conv_outputs.append(h_reshape)

        # Concatenate the outputs from different filters
        if len(self.filter_sizes) > 1:
            rnn_inputs = tf.concat(conv_outputs, -1)
        else:
            rnn_inputs = h_reshape

        sequence_length = rnn_inputs.shape[1]
        dim = rnn_inputs.shape[2]
        # LSTM cell
        rnn_inputs = tf.transpose(rnn_inputs, [1, 0, 2])
        rnn_inputs = tf.reshape(rnn_inputs, [-1, dim])
        rnn_inputs = tf.split(rnn_inputs, sequence_length, 0)
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to LSTM cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        self._initial_state.append(cell.zero_state(self.batch_size, dtype=tf.float32))

        # Feed the CNN outputs to LSTM network
        with tf.variable_scope('LSTM_%d' % num):
            outputs, state = tf.nn.static_rnn(cell,
                                              rnn_inputs,
                                              dtype=tf.float32)
            # self.final_state = state

        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        outputs = self.attention(outputs, self.attn_size, num)

        return outputs


    @staticmethod
    def attention(inputs, attention_size, num):
        """
            Attention mechanism layer.
            :param inputs: outputs of RNN/Bi-RNN layer (not final state)
            :param attention_size: linear size of attention weights
            :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
            """
        # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
        if isinstance(inputs, tuple):
            inputs = tf.concat(2, inputs)

        sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.get_variable("W_omega_%d" % num, initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.get_variable("b_omega_%d" % num, initializer=tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.get_variable("u_omega_%d" % num, initializer=tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        # if l2_reg_lambda > 0:
        #    l2_loss += tf.nn.l2_loss(W_omega)
        #    l2_loss += tf.nn.l2_loss(b_omega)
        #    l2_loss += tf.nn.l2_loss(u_omega)
        #    tf.add_to_collection('losses', l2_loss)

        return output

    @staticmethod
    def get_embding(file_name, embedding_size):
        with open(file_name, "rb") as f:
            d = pickle.load(f)
        d[0] = [0] * embedding_size
        return d
