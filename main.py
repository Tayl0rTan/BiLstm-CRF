# coding=utf-8

import numpy as np
import tensorflow as tf
import time
from data_util import load_data
import os
from modeling import DAModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, labels, batch_size):
    data_size = len(data)
    num_batches_per_epoch = int((len(data) + batch_size - 1) / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index: end_index], labels[start_index: end_index]


batchSize = 16
log_dir = "log"
model_dir = "DAModel"
model_name = "ckpt"

def main():
    train_data, train_labels = load_data('data/ner_train.txt')
    dev_data, dev_labels = load_data('data/ner_test.txt')
    print(len(train_data),len(dev_data))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:
        model = DAModel()
        sess.run(tf.global_variables_initializer())
        clip = 2
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("train-log", sess.graph)
        counter = 0
        for epoch in range(1000):
            for dialogues, labels in minibatches(train_data, train_labels, batchSize):
                _, dialogue_lengthss = pad_sequences(dialogues, 0)
                word_idss, utterance_lengthss = pad_sequences(dialogues, 0, nlevels=2)
                true_labs = labels
                labs_t, _ = pad_sequences(true_labs, 0)
                counter += 1
                train_loss, train_accuracy, _ = sess.run([model.loss, model.accuracy, model.train_op],
                                                         feed_dict={model.word_ids: word_idss,
                                                                    model.utterance_lengths: utterance_lengthss,
                                                                    model.dialogue_lengths: dialogue_lengthss,
                                                                    model.labels: labs_t, model.clip: clip})
                # writer.add_summary(summary, global_step = counter)
                print("step = {}, train_loss = {}, train_accuracy = {}".format(counter, train_loss, train_accuracy))

                train_precision_summ = tf.Summary()
                train_precision_summ.value.add(tag='train_accuracy', simple_value=train_accuracy)
                writer.add_summary(train_precision_summ, counter)

                train_loss_summ = tf.Summary()
                train_loss_summ.value.add(tag='train_loss', simple_value=train_loss)
                writer.add_summary(train_loss_summ, counter)
                if counter % 100 == 0:
                    loss_dev = []
                    acc_dev = []
                    for dialogues, labels in minibatches(dev_data, dev_labels, batchSize):
                        _, dialogue_lengthss = pad_sequences(dialogues, 0)
                        word_idss, utterance_lengthss = pad_sequences(dialogues, 0, nlevels=2)
                        true_labs = labels
                        labs_t, _ = pad_sequences(true_labs, 0)
                        dev_loss, dev_accuacy = sess.run([model.loss, model.accuracy],
                                                         feed_dict={model.word_ids: word_idss,
                                                                    model.utterance_lengths: utterance_lengthss,
                                                                    model.dialogue_lengths: dialogue_lengthss,
                                                                    model.labels: labs_t})
                        loss_dev.append(dev_loss)
                        acc_dev.append(dev_accuacy)
                    valid_loss = sum(loss_dev) / len(loss_dev)
                    valid_accuracy = sum(acc_dev) / len(acc_dev)

                    dev_precision_summ = tf.Summary()
                    dev_precision_summ.value.add(tag='dev_accuracy', simple_value=valid_accuracy)
                    writer.add_summary(dev_precision_summ, counter)

                    dev_loss_summ = tf.Summary()
                    dev_loss_summ.value.add(tag='dev_loss', simple_value=valid_loss)
                    writer.add_summary(dev_loss_summ, counter)
                    print("counter = {}, dev_loss = {}, dev_accuacy = {}".format(counter, valid_loss, valid_accuracy))
                    saver.save(sess,'model/model.ckpt')


if __name__ == "__main__":
    main()
