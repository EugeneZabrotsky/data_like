#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import config

DATA_PATH = config.get_data_path()

CUST_TRAIN_FILE = DATA_PATH / 'customer_train.csv'
CUST_TEST_FILE = DATA_PATH / 'customer_test.csv'

CUST_TRAIN_SAVE = DATA_PATH / 'customer_train_processed.csv'
CUST_TEST_SAVE = DATA_PATH / 'customer_test_processed.csv'


def process(cust):
    mar_statuses = cust['marital_status_cd'].unique()
    mar2id = dict(zip(mar_statuses, range(len(mar_statuses))))

    cust['marital_status_cd'] = cust['marital_status_cd'].map(mar2id)
    cust['gender_cd'] = cust['gender_cd'].map({'M': 0, 'F': 1})

    for i in range(7):
        cust[f'product_{i}'] = cust[f'product_{i}'].map({'OPN': 1, 'UTL': 2, 'CLS': 3})

    prod_cols = [f'product_{i}' for i in range(7)]
    cust['prod_not_nan'] = (cust[prod_cols] > 0).sum(axis=1)
    cust['prod_sum_opn'] = (cust[prod_cols] == 1).sum(axis=1)
    cust['prod_sum_utl'] = (cust[prod_cols] == 2).sum(axis=1)
    cust['prod_sum_cls'] = (cust[prod_cols] == 3).sum(axis=1)
    
    return cust


if __name__ == '__main__':
    cust_train = pd.read_csv(CUST_TRAIN_FILE)
    cust_test = pd.read_csv(CUST_TEST_FILE)

    cust_train_processed = process(cust_train)
    cust_test_processed = process(cust_test)

    cust_train_processed.to_csv(CUST_TRAIN_SAVE)
    cust_test_processed.to_csv(CUST_TEST_SAVE)

    