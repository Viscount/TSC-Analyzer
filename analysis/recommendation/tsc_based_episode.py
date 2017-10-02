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
from sklearn import tree
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


def get_episodes():
    episode_set = set()
    episodes = episode_dao.find_all_episodes()
    for episode in episodes:
        episode_set.add(episode.episode_id)
    return episode_set


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
    def __init__(self, episode_dict):
        self.episode_dict = episode_dict

    def __iter__(self):
        for episode in self.episode_dict.keys():
            content_words = self.episode_dict[episode]
            yield TaggedDocument(words=content_words, tags=[episode])


def split_dataset(watched_vector, ratio):
    random.seed(123456789)
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


def make_episode_words(episode_id, model):
    words = []
    danmakus = danmaku_dao.find_danmakus_by_episode(episode_id)
    for danmku in danmakus:
        words.extend(preprocess_util.word_segment(danmku.content))
    for word in words:
        if word not in model.wv.vocab:
            words.remove(word)
    return model[words]


def model_training():
    episodes = episode_dao.find_all_episodes()
    episode_dict = dict()
    for episode in episodes:
        print 'Collecting episode ' + str(episode.episode_id)
        words = []
        danmakus = danmaku_dao.find_danmakus_by_episode(episode.episode_id)
        for danmku in danmakus:
            words.extend(preprocess_util.word_segment(danmku.content))
        episode_dict[str(episode.episode_id)] = words
    tsc_docs = TscTaggedDocument(episode_dict)
    model = Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=10, iter=10, workers=multiprocessing.cpu_count())
    print 'Building vocabulary......'
    model.build_vocab(tsc_docs)
    print 'Training doc2vec model......'
    model.train(tsc_docs, total_examples=model.corpus_count, epochs=model.iter)
    print 'Vocabulary size:', len(model.wv.vocab)
    model.save("D:\\workspace\\TSC-Analyzer\\tmp\\models\\episode_doc2vec_200.model")


def transfer(origin_result, test_set, times_limit, lookup):
    bangumi_list = []
    index = 0
    for bangumi_index in test_set:
        current_episode_watch_count = 0
        season_id = lookup[str(bangumi_index)]
        episodes = episode_dao.find_episodes_by_bangumi(season_id)
        for episode in episodes:
            if origin_result[index] > 0:
                current_episode_watch_count += 1
            index += 1
        if current_episode_watch_count >= times_limit:
            bangumi_list.append(1)
        else:
            bangumi_list.append(0)
    return np.array(bangumi_list)


if __name__ == "__main__":
    model_training()

    # senders = get_senders("D:\\workspace\\TSC-Analyzer\\tmp\\senders.csv", 500)
    # bangumis = get_bangumis()
    # episodes = get_episodes()
    # id_user_lookup, user_id_lookup = make_lookup(senders)
    # id_bangumi_lookup, bangumi_id_lookup = make_lookup(bangumis)
    # id_episode_lookup, episode_id_lookup = make_lookup(episodes)
    #
    # matrix = np.loadtxt("matrix.txt", delimiter=",")
    # matrix_epi = np.loadtxt("matrix_user2episode.txt", delimiter=",")
    # sender_count = matrix.shape[0]
    # bangumi_count = matrix.shape[1]
    # episode_count = matrix_epi.shape[1]
    # model = Doc2Vec.load("D:\\workspace\\TSC-Analyzer\\tmp\\models\\content_doc2vec_200.model")
    #
    # # tsc-based
    # ratio = 0.8
    # result_list = []
    # for index in range(0, sender_count):
    #     bangumi_watch = matrix[index, :]
    #     episode_watch = matrix_epi[index, :]
    #     train_set, test_set = split_dataset(bangumi_watch, ratio)
    #     if train_set is None and test_set is None:
    #         continue
    #     # build expand train_set
    #     train_list = []
    #     label_list = []
    #     for bangumi in train_set:
    #         season_id = str(id_bangumi_lookup[str(bangumi)])
    #         episodes = episode_dao.find_episodes_by_bangumi(season_id)
    #         for episode in episodes:
    #             train_list.append(make_episode_words(episode.episode_id, model))
    #             label_list.append(episode_watch[episode_id_lookup[str(episode.episode_id)]])
    #     x_train = np.array(train_list)
    #     y_train = np.array(label_list)
    #
    #     # build expand test_set
    #     test_list = []
    #     label_list = []
    #     for bangumi in test_set:
    #         season_id = str(id_bangumi_lookup[str(bangumi)])
    #         episodes = episode_dao.find_episodes_by_bangumi(season_id)
    #         for episode in episodes:
    #             test_list.append(make_episode_words(episode.episode_id, model))
    #             label_list.append(episode_watch[episode_id_lookup[str(episode.episode_id)]])
    #     x_predict = np.array(test_list)
    #     y_predict = np.array(label_list)
    #     y_bangumi_predict = np.array(list(bangumi_watch[watched_index] for watched_index in test_set))
    #
    #     gnb = GaussianNB()
    #     gnb = gnb.fit(x_train, y_train)
    #     result = gnb.predict(x_predict)
    #     bangumi_result = transfer(result, test_set, 1, id_bangumi_lookup)
    #
    #     precision = precision_score(y_bangumi_predict, bangumi_result)
    #     recall = recall_score(y_bangumi_predict, bangumi_result)
    #     print "No.%d pre=%.2f rec=%.2f" % (index, precision, recall)
    #     result_list.append((precision, recall))
    #
    # overall = pd.DataFrame(result_list, columns=["Precision", "Recall"])
    # print "Avg precision: %.2f Avg recall: %.2f" % (overall["Precision"].mean(), overall["Recall"].mean())



