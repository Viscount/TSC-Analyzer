#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from dao import danmaku_dao

if __name__ == "__main__":
    user_list = danmaku_dao.find_all_senders()
    print len(user_list)
