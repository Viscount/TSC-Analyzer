#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import pandas as pd

if __name__ == "__main__":
    overall = pd.DataFrame.from_csv("tsc-episode-mlp.csv", header=0)
    sorted = overall.sort_values(["Precision"], ascending=False).head(500)
    print "Avg accuracy: %.4f Avg precision: %.4f Avg recall: %.4f" % \
          (sorted["Accuracy"].mean(), sorted["Precision"].mean(), sorted["Recall"].mean())
