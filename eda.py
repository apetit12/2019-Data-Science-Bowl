#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: antoinepetit
"""

import pandas as pd

# LOAD DATA
file_name_train = 'Data/train.csv'
file_name_label = 'Data/train_labels.csv'
df = pd.read_csv(file_name_train)
df_labels = pd.read_csv(file_name_label)

