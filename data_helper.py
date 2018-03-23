# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import time
import json
import collections

import numpy as np
from tensorflow.contrib import learn


def load_data(file_path, sw_path=None, min_frequency=0, max_length=0, language='ch', vocab_processor=None, shuffle=True):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print('Building dataset ...')
        start = time.time()

        labels = list()
        data = list()
        label_map = {
            0: 0,
            2: 1,
            5: 2,
            20: 3,
            22: 4,
            25: 5,
            39: 6,
        }
        for line in f:
            item = line.split(",")
            label = int(item[0])
            # label = label_map[label]
            # tmp = [0] * 7
            # tmp[label] = 1
            labels.append(label)
            index_list = [int(index) for index in item[1].split(" ")[:-1]]
            if len(index_list) < max_length:
                index_list.extend([0]*(max_length - len(index_list)))
            data.append(index_list)

    labels = np.array(labels)
    data = np.array(data)

    data_size = len(data)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))

    return data, labels


def batch_iter(union_data, labels, batch_size, num_epochs, max_len, max_char_len):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param union_data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(union_data) == len(labels)

    data_size = len(union_data)
    epoch_length = data_size // batch_size

    for _ in range(0, num_epochs):
        print("-----------current epochs: %d-----------" % _)
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xchardata = list()
            xdata = list()
            for j in range(start_index, end_index):
                xdata.append(union_data[j][0])
                xchardata.append(union_data[j][1])
            ydata = labels[start_index: end_index]
            sequence_length = [max_len] * len(xdata)
            char_sequence_length = [max_char_len] * len(xdata)

            yield xdata, xchardata, ydata, sequence_length, char_sequence_length

# --------------- Private Methods ---------------

def _tradition_2_simple(sent):
    """ Convert Traditional Chinese to Simplified Chinese """
    # Please download langconv.py and zh_wiki.py first
    # langconv.py and zh_wiki.py are used for converting between languages
    try:
        import langconv
    except ImportError as e:
        error = "Please download langconv.py and zh_wiki.py at "
        error += "https://github.com/skydark/nstools/tree/master/zhtools."
        print(str(e) + ': ' + error)
        sys.exit()

    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent):
    """ Tokenizer for Chinese """
    import jieba
    sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
    return re.sub(r'\s+', ' ', sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
        sent = re.sub('！{2,}', '！', sent)
        sent = re.sub('？{2,}', '！', sent)
        sent = re.sub('。{2,}', '。', sent)
        sent = re.sub('，{2,}', '，', sent)
        sent = re.sub('\s{2,}', ' ', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    if sw is not None:
        sent = "".join([word for word in sent if word not in sw])

    return sent