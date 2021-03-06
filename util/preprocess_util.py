#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import jieba.posseg as segtool
import re

ACCEPTABLE_TYPE = {'n', 't', 's', 'f', 'v', 'a', 'b', 'z', 'e', 'y', 'o'}
REPLACE_DICT = {
    "233+": "233",
    "666+": "666"
}


def check_type(word_type):
    if word_type[0] in ACCEPTABLE_TYPE:
        return True
    else:
        return False


def check_replace(word):
    for item in REPLACE_DICT.keys():
        pattern = re.compile(item)
        if re.match(pattern, word) is not None:
            new_word = REPLACE_DICT[item]
            return new_word
    return word


def word_segment(content):
    words = []
    results = segtool.cut(content)
    for result in results:
        result.word = check_replace(result.word)
        if check_type(result.flag):
            words.append(result.word)
    return words

if __name__ == "__main__":
    words = word_segment("士郎这是要搞基？")
    print [word for word in words]
