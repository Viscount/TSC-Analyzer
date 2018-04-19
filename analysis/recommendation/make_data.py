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


def get_sender_episode(sender_id):
    danmakus = danmaku_dao.find_danmakus_by_sender_id(sender_id)
    episode_id_set = set()
    for danmaku in danmakus:
        episode_id_set.add(danmaku.episode_id)
    return episode_id_set


def get_sender_history_detail(sender_ids):
    history = []
    danmakus = danmaku_dao.find_danmakus_by_sender_ids(sender_ids)
    episodes = episode_dao.find_all_episodes()
    epi_bangumi_dict = dict()
    for episode in episodes:
        epi_bangumi_dict[episode.episode_id] = episode.season_id
    for danmaku in danmakus:
        item = [danmaku.sender_id, epi_bangumi_dict[danmaku.episode_id], danmaku.episode_id,
                danmaku.raw_id, danmaku.unix_timestamp]
        history.append(item)
    return history


if __name__ == "__main__":
    senders = get_senders("D:\\workspace\\TSC-Analyzer\\tmp\\senders.csv", 248)
    bangumis = get_bangumis()
    episodes = get_episodes()
    id_user_lookup, user_id_lookup = make_lookup(senders)
    id_bangumi_lookup, bangumi_id_lookup = make_lookup(bangumis)
    id_episode_lookup, episode_id_lookup = make_lookup(episodes)

    # matrix = np.zeros((len(senders), len(bangumis)))
    # for sender in senders:
    #    sender_index = int(user_id_lookup[sender])
    #    bangumis_set = get_sender_bangumi(sender)
    #    for bangumi in bangumis_set:
    #        bangumi_index = int(bangumi_id_lookup[bangumi])
    #        matrix[sender_index][bangumi_index] = 1
    # np.savetxt("matrix_user2bangumi.txt", matrix, fmt="%d", delimiter=",")

    # matrix = np.zeros((len(senders), len(episodes)))
    # for sender in senders:
    #     sender_index = int(user_id_lookup[sender])
    #     episode_set = get_sender_episode(sender)
    #     for episode in episode_set:
    #         episode_index = int(episode_id_lookup[episode])
    #         matrix[sender_index][episode_index] = 1
    # np.savetxt("matrix_user2episode.txt", matrix, fmt="%d", delimiter=",")

    with open("D:\\workspace\\TSC-Analyzer\\tmp\\sender_his.csv", 'wb') as f:
        writer = csv.writer(f, dialect='excel', delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["sender_id", "seanson_id", "episode_id", "tsc_raw_id", "post_time"])
        history = get_sender_history_detail(senders)
        for item in history:
            writer.writerow(item)
