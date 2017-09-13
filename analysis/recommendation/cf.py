#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import csv
from dao import bangumi_dao, episode_dao, danmaku_dao
import numpy as np


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


if __name__ == "__main__":
    senders = get_senders("D:\\workspace\\TSC-Analyzer\\tmp\\senders.csv", 500)
    bangumis = get_bangumis()
    id_user_lookup, user_id_lookup = make_lookup(senders)
    id_bangumi_lookup, bangumi_id_lookup = make_lookup(bangumis)
    matrix = np.zeros((len(senders), len(bangumis)))
    for sender in senders:
        sender_index = int(user_id_lookup[sender])
        bangumis_set = get_sender_bangumi(sender)
        for bangumi in bangumis_set:
            bangumi_index = int(bangumi_id_lookup[bangumi])
            matrix[sender_index][bangumi_index] = 1
    np.savetxt("matrix.txt", matrix, fmt="%d", delimiter=",")



