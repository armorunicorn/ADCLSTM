# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import json
import datetime
import pickle as pkl

import sklearn
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

import data_helper
# from rnn_classifier import rnn_clf
# from cnn_classifier import cnn_clf
# from attn_rnn_classifier import attn_rnn_clf
# from clstm_classifier import clstm_clf
from double_aclstm_classifier import double_aclstm_clf as clstm_clf

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    error = "Please install scikit-learn."
    print(str(e) + ': ' + error)
    sys.exit()

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# =============================================================================

# Model choices
tf.flags.DEFINE_string('clf', 'adclstm', "Type of classifiers. Default: cnn. You have four choices: [cnn, lstm, blstm, clstm, adclstm]")
tf.flags.DEFINE_string('embd_file', './data/word_embd.pickle', "word embeding pickle file")
tf.flags.DEFINE_string('char_embd_file', './data/char_embd.pickle', "Char embeding pickle file")

# Data parameters
tf.flags.DEFINE_string('data_file', "./data/model_input_word_data.txt", 'Data file path')
tf.flags.DEFINE_string('char_data_file', "./data/model_input_char_data.txt", 'Char Data file path')
tf.flags.DEFINE_string('stop_word_file', None, 'Stop word file path')
tf.flags.DEFINE_string('language', 'en', "Language of the data file. You have two choices: [ch, en]")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 12, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 28, 'Max document length')
tf.flags.DEFINE_integer('char_max_length', 51, 'Char Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0.2, 'Cross validation test size')

# Model hyperparameters
tf.flags.DEFINE_integer('embedding_size', 200, 'Word embedding size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('char_embedding_size', 200, 'Word embedding size. For CNN, C-LSTM.')
tf.flags.DEFINE_string('filter_sizes', '3, 4, 5', 'CNN filter sizes. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('hidden_size', 100, 'Number of hidden units in the LSTM cell. For LSTM, Bi-LSTM')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM')
tf.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')  # All
tf.flags.DEFINE_float('l2_reg_lambda', 0.001, 'L2 regularization lambda')  # All

tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_integer('evaluate_every_steps', 100, 'Evaluate the model on validation set after this many steps')
tf.flags.DEFINE_integer('save_every_steps', 4000, 'Save the model after this many steps')
tf.flags.DEFINE_integer('num_checkpoint', 10, 'Number of models to store')

FLAGS = tf.flags.FLAGS

if FLAGS.clf == 'lstm':
    FLAGS.embedding_size = FLAGS.hidden_size
elif FLAGS.clf == 'clstm' or FLAGS.clf == 'adclstm':
    FLAGS.hidden_size = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters

# Output files directory
timestamp = str(int(time.time()))
outdir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Load and save data
# =============================================================================

data, labels = data_helper.load_data(file_path=FLAGS.data_file,
                                                               sw_path=FLAGS.stop_word_file,
                                                               min_frequency=FLAGS.min_frequency,
                                                               max_length=FLAGS.max_length,
                                                               language=FLAGS.language,
                                                               shuffle=True)

char_data, char_labels = data_helper.load_data(file_path=FLAGS.char_data_file,
                                                               sw_path=FLAGS.stop_word_file,
                                                               min_frequency=FLAGS.min_frequency,
                                     max_length=FLAGS.char_max_length,
                                                               language=FLAGS.language,
                                                           shuffle=True)

union_data = list()
for i in range(0, len(data)):
    tmp = [data[i], char_data[i]]
    union_data.append(tmp)

params = FLAGS.__flags
# Print parameters
model = params['clf']
if model == 'cnn':
    del params['hidden_size']
    del params['num_layers']
elif model == 'lstm' or model == 'blstm':
    del params['num_filters']
    del params['filter_sizes']
    params['embedding_size'] = params['hidden_size']
elif model == 'clstm' or model == 'adclstm':
    params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']

params_dict = sorted(params.items(), key=lambda x: x[0])
print('Parameters:')
for item in params_dict:
    print('{}: {}'.format(item[0], item[1]))
print('')

# Save parameters to file
params_file = open(os.path.join(outdir, 'params.pkl'), 'wb')
pkl.dump(params, params_file, True)
params_file.close()


# Simple Cross validation
# TODO use k-fold cross validation
x_train, x_valid, y_train, y_valid = train_test_split(union_data,
                                                    labels,
                                                    test_size=FLAGS.test_size,
                                                    random_state=0)
# Batch iterator
train_data = data_helper.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs, FLAGS.max_length,
                                    FLAGS.char_max_length)

# Train
# =============================================================================

with tf.Graph().as_default():
    # config = tf.ConfigProto()
    # config.gpu_options.allocator_type = 'BFC'
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        if FLAGS.clf == 'cnn':
            classifier = cnn_clf(FLAGS)
        elif FLAGS.clf == 'lstm' or FLAGS.clf == 'blstm':
            classifier = rnn_clf(FLAGS)
        elif FLAGS.clf == 'clstm' or FLAGS.clf == 'adclstm':
            classifier = clstm_clf(FLAGS)
        elif FLAGS.clf == "attn_lstm":
            classifier = attn_rnn_clf(FLAGS)
        else:
            raise ValueError('clf should be one of [cnn, lstm, blstm, clstm, adclstm]')

        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(classifier.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Summaries
        loss_summary = tf.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)

        # Train summary
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summary
        valid_summary_op = tf.summary.merge_all()
        valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoint)

        sess.run(tf.global_variables_initializer())


        def run_step(input_data, is_training=True):
            """Run one step of the training process."""
            input_x, input_char_x, input_y, sequence_length, char_sequence_length = input_data

            fetches = {'step': global_step,
                       'cost': classifier.cost,
                       'accuracy': classifier.accuracy}
            feed_dict = {classifier.input_x: input_x,
                         classifier.input_char_x: input_char_x,
                         classifier.input_y: input_y}

            fetches['correct_num'] = classifier.correct_num
            fetches['predictions'] = classifier.predictions
            # fetches['input_y'] = classifier.input_y
            if FLAGS.clf != 'cnn':
                fetches['final_state'] = classifier.final_state
                feed_dict[classifier.batch_size] = len(input_x)
                feed_dict[classifier.sequence_length] = sequence_length
                feed_dict[classifier.char_sequence_length] = char_sequence_length

            if is_training:
                fetches['train_op'] = train_op
                fetches['summaries'] = train_summary_op
                feed_dict[classifier.keep_prob] = FLAGS.keep_prob
            else:
                fetches['summaries'] = valid_summary_op

                feed_dict[classifier.keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            accuracy = vars['accuracy']
            summaries = vars['summaries']
            correct_num = vars['correct_num']
            predictions = vars['predictions']
            # correct_num = vars['correct_num']

            # Write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                valid_summary_writer.add_summary(summaries, step)

            time_str = datetime.datetime.now().isoformat()
            if is_training:
                print("{}: step: {}, loss: {:g}, accuracy: {:g}, num: {}".format(time_str, step, cost, accuracy, correct_num))

            return correct_num, predictions, input_y


        print('Start training ...')

        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)

            #if current_step % FLAGS.evaluate_every_steps == 0:


            # if current_step % FLAGS.save_every_steps == 0:
            #    save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)
        print('\nValidation')
        valid_data = data_helper.batch_iter(x_valid, y_valid, FLAGS.batch_size, 1, FLAGS.max_length, FLAGS.char_max_length)
        correct_num = 0
        total_predictions = list()
        total_input_y = list()
        for valid_input in valid_data:
            cur_correct_num, predictions, input_y = run_step(valid_input, is_training=False)
            correct_num += cur_correct_num
            total_predictions.extend(predictions)
            total_input_y.extend(input_y)
        correct_num /= len(x_valid)
        recall = sklearn.metrics.recall_score(total_input_y, total_predictions, average="macro")
        print('END:%g' % correct_num)
        print("recall:%g" % recall)

        saver.save(sess, os.path.join(outdir, 'model/clf'), 1)
        print('\nAll the files have been saved to {}\n'.format(outdir))
