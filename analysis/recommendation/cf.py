#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import csv
from dao import bangumi_dao


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
    return id_item_lookup, item_id_lookup


def get_sender_bangumi():
    pass


if __name__ == "__main__":
    senders = get_senders("D:\\workspace\\TSC-Analyzer\\tmp\\senders.csv", 100)
    bangumis = get_bangumis()
    id_user_lookup, user_id_lookup = make_lookup(senders)
    id_bangumi_lookup, bangumi_id_lookup = make_lookup(bangumis)
    


