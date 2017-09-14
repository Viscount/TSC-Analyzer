#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import csv
from dao import bangumi_dao, episode_dao, danmaku_dao
import numpy as np
import pandas as pd
from numpy import linalg


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


def evaluate(standard, predict):
    dim = standard.shape[0]
    std_rank = np.argsort(-standard)
    predict_rank = np.argsort(-predict)
    std_result_set = set()
    predict_set = set()
    for index in range(0, dim):
        if standard[std_rank[index]] == 0:
            break
        else:
            std_result_set.add(std_rank[index])
            predict_set.add(predict_rank[index])
    return len(std_result_set & predict_set) * 1.0 / index


if __name__ == "__main__":
    # senders = get_senders("D:\\workspace\\TSC-Analyzer\\tmp\\senders.csv", 500)
    # bangumis = get_bangumis()
    # id_user_lookup, user_id_lookup = make_lookup(senders)
    # id_bangumi_lookup, bangumi_id_lookup = make_lookup(bangumis)
    # matrix = np.zeros((len(senders), len(bangumis)))
    # for sender in senders:
    #     sender_index = int(user_id_lookup[sender])
    #     bangumis_set = get_sender_bangumi(sender)
    #     for bangumi in bangumis_set:
    #         bangumi_index = int(bangumi_id_lookup[bangumi])
    #         matrix[sender_index][bangumi_index] = 1
    # np.savetxt("matrix.txt", matrix, fmt="%d", delimiter=",")

    matrix = np.loadtxt("matrix.txt", delimiter=",")
    sender_count = matrix.shape[0]
    bangumi_count = matrix.shape[1]
    # user-based cf
    result_list = []
    for sender_index in range(0, sender_count):
        standard = matrix[sender_index, :]
        score = np.zeros(bangumi_count)
        # 获取相邻用户
        compare_user_set = set()
        for bangumi_index in range(0, bangumi_count):
            if standard[bangumi_index] == 1:
                bangumi_watched = matrix[:, bangumi_index]
                for user_index in range(0, sender_count):
                    if bangumi_watched[user_index] == 1 & user_index != sender_index:
                        compare_user_set.add(user_index)
        # 计算推荐值
        for user_index in compare_user_set:
            user_array = matrix[user_index, :]
            similarity = sim(standard, user_array)
            for index in range(0, bangumi_count):
                score[index] += user_array[index] * similarity
        # 计算评估结果
        evl = evaluate(standard, score)
        print evl
        result_list.append(evl)
    overall = pd.Series(result_list)
    print "Total: %f" % (overall.mean())






