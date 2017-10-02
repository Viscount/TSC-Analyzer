#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import csv
from dao import bangumi_dao, episode_dao, danmaku_dao
import numpy as np
import pandas as pd
import math
import random
from numpy import linalg
from sklearn.metrics import precision_score, recall_score


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


def sim(a, b):
    num = np.dot(a, b)  # 若为行向量则 A * B.T
    denom = linalg.norm(a) * linalg.norm(b)
    cos = num / denom  # 余弦值
    return cos


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


def norm(score_array):
    score_sum = np.max(score_array)
    normlized = score_array / score_sum
    for index in range(0, normlized.shape[0]):
        if normlized[index] > 0.5:
            normlized[index] = 1
        else:
            normlized[index] = 0
    return normlized


if __name__ == "__main__":

    matrix = np.loadtxt("matrix.txt", delimiter=",")
    sender_count = matrix.shape[0]
    bangumi_count = matrix.shape[1]
    # user-based cf
    result_list = []
    ratio = 0.8
    for sender_index in range(0, sender_count):
        bangumi_watched = matrix[sender_index, :]
        train_set, test_set = split_dataset(bangumi_watched, ratio)
        if train_set is None and test_set is None:
            continue
        x_train = np.array(list(bangumi_watched[bangumi_index] for bangumi_index in train_set))
        x_predict = np.array(list(bangumi_watched[bangumi_index] for bangumi_index in test_set))
        score = np.zeros(len(test_set))
        for sender_compare in range(0, sender_count):
            if sender_compare != sender_index:
                comp_watched = matrix[sender_compare, :]
                comp_train = np.array(list(comp_watched[bangumi_index] for bangumi_index in train_set))
                similarity = sim(x_train, comp_train)
                if similarity > 0:
                    comp_predict = np.array(list(comp_watched[bangumi_index] for bangumi_index in test_set))
                    score += similarity * comp_predict
        result = norm(score)
        precision = precision_score(x_predict, result)
        recall = recall_score(x_predict, result)
        print "No.%d pre=%.2f rec=%.2f" % (sender_index, precision, recall)
        result_list.append((precision, recall))

    overall = pd.DataFrame(result_list, columns=["Precision", "Recall"])
    print "Avg precision: %.2f Avg recall: %.2f" % (overall["Precision"].mean(), overall["Recall"].mean())


    # item-based cf
    '''sim_bangumi = np.zeros((bangumi_count, bangumi_count))
    result_list = []
    for index in range(0, bangumi_count):
        for index_com in range(index, bangumi_count):
            if index == index_com:
                sim_bangumi[index, index_com] = 1.0
            else:
                similarity = item_sim(matrix[:, index], matrix[:, index_com])
                sim_bangumi[index, index_com] = similarity
                sim_bangumi[index_com, index] = similarity
    ratio = 0.8
    for index in range(0, sender_count):
        bangumi_watch = matrix[index, :]
        score = np.zeros(bangumi_count)
        train_set, test_set = split_watched(bangumi_watch, ratio)
        if len(train_set) == 0 or len(test_set) == 0:
            continue
        for train_item in train_set:
            for bangumi_id in range(0, bangumi_count):
                if bangumi_id != train_item:
                    score[bangumi_id] += sim_bangumi[train_item, bangumi_id]
        evl = evaluate_item_based(train_set, test_set, score)
        print evl
        result_list.append(evl)
    overall = pd.Series(result_list)
    print "Total: %f" % (overall.mean())'''






