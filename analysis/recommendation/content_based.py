#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import csv
from dao import bangumi_dao, episode_dao, danmaku_dao
import numpy as np
import pandas as pd
import random
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import multiprocessing
from util import preprocess_util
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger("logger")
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def get_senders(path, limit):
    senders = set()
    csv_reader = csv.reader(open(path, mode="r"))
    for row in csv_reader:
        content = row[0].split("\t")
        sender_id = content[0]
        num = int(content[1])
        if num < limit:
            break
        else:
            senders.add(sender_id)
    return senders


def get_bangumis():
    bangumi_set = set()
    bangumis = bangumi_dao.find_all_bangumis()
    for bangumi in bangumis:
        bangumi_set.add(bangumi.season_id)
    return bangumi_set


def make_lookup(item_set):
    id_item_lookup = dict()
    item_id_lookup = dict()
    index = 0
    for item in item_set:
        id_item_lookup[str(index)] = item
        item_id_lookup[item] = str(index)
        index += 1
    return id_item_lookup, item_id_lookup


def get_sender_bangumi(sender_id):
    danmakus = danmaku_dao.find_danmakus_by_sender_id(sender_id)
    episode_id_set = set()
    for danmaku in danmakus:
        episode_id_set.add(danmaku.episode_id)
    episodes = episode_dao.find_episodes_by_ids(list(episode_id_set))
    bangumis_set = set()
    for episode in episodes:
        bangumis_set.add(episode.season_id)
    return bangumis_set


class TscTaggedDocument(object):
    def __init__(self, bangumis):
        self.bangumis = bangumis

    def __iter__(self):
        for bangumi in self.bangumis:
            intro_words = preprocess_util.word_segment(bangumi.introduction)
            yield TaggedDocument(words=intro_words, tags=[str(bangumi.season_id)])


def model_training():
    bangumis = bangumi_dao.find_all_bangumis()
    tsc_docs = TscTaggedDocument(bangumis)
    model = Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=1, iter=10, workers=multiprocessing.cpu_count())
    print 'Building vocabulary......'
    model.build_vocab(tsc_docs)
    print 'Training doc2vec model......'
    model.train(tsc_docs, total_examples=model.corpus_count, epochs=model.iter)
    print 'Vocabulary size:', len(model.wv.vocab)
    model.save("intro_doc2vec_200.model")


def split_dataset(watched_vector, ratio):
    watched_set = set()
    unwatched_set = set()
    for index in range(0, watched_vector.shape[0]):
        if watched_vector[index] > 0:
            watched_set.add(index)
        else:
            unwatched_set.add(index)
    total_select_num = int(watched_vector.shape[0] * ratio)
    watched_select_num = int(len(watched_set) * ratio)
    if watched_select_num == len(watched_set) or watched_select_num == 0:
        return None, None
    watched_select = set(random.sample(list(watched_set), watched_select_num))
    unwatched_select = set(random.sample(list(unwatched_set), total_select_num - watched_select_num))
    train_set = watched_select | unwatched_select
    test_set = (watched_set - watched_select) | (unwatched_set - unwatched_select)
    return train_set, test_set


if __name__ == "__main__":
    # model_training()

    senders = get_senders("D:\\workspace\\TSC-Analyzer\\tmp\\senders.csv", 500)
    bangumis = get_bangumis()
    id_user_lookup, user_id_lookup = make_lookup(senders)
    id_bangumi_lookup, bangumi_id_lookup = make_lookup(bangumis)

    matrix = np.loadtxt("matrix.txt", delimiter=",")
    sender_count = matrix.shape[0]
    bangumi_count = matrix.shape[1]
    model = Doc2Vec.load("intro_doc2vec_200.model")

    # content-based
    ratio = 0.8
    result_list = []
    for index in range(0, sender_count):
        bangumi_watch = matrix[index, :]
        score = np.zeros(bangumi_count)
        train_set, test_set = split_dataset(bangumi_watch, ratio)
        if train_set is None and test_set is None:
            continue
        # build dataset
        x_train = np.array(list(model.docvecs[str(id_bangumi_lookup[str(bangumi_id)])] for bangumi_id in train_set))
        x_predict = np.array(list(model.docvecs[str(id_bangumi_lookup[str(bangumi_id)])] for bangumi_id in test_set))
        y_train = np.array(list(bangumi_watch[watched_index] for watched_index in train_set))
        y_predict = np.array(list(bangumi_watch[watched_index] for watched_index in test_set))
        gnb = GaussianNB()
        result = gnb.fit(x_train, y_train).predict(x_predict)
        precision = precision_score(y_predict, result)
        recall = recall_score(y_predict, result)
        # total = x_predict.shape[0]
        # miss = (y_predict != result).sum()

        print "No.%d pre=%.2f rec=%.2f" % (index, precision, recall)
        result_list.append((precision, recall))
    overall = pd.DataFrame(result_list, columns=["Precision", "Recall"])
    print "Avg precision: %.2f Avg recall: %.2f" % (overall["Precision"].mean(), overall["Recall"].mean())



