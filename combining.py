#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import config

DATA_PATH = config.get_data_path()

STORIES_FILE = DATA_PATH / 'stories_features.csv'
REACTION_TRAIN_FILE = DATA_PATH / 'reactions_features_train.csv'
REACTION_TEST_FILE = DATA_PATH / 'reactions_features_test.csv'

CUSTOMER_TRAIN_FILE = DATA_PATH / 'customer_features_train.csv'
CUSTOMER_TEST_FILE = DATA_PATH / 'customer_features_test.csv'

TRANSACTIONS_FILE = DATA_PATH / 'transactions_features.csv'

TRAIN_SAVE = DATA_PATH / 'train_features.csv'
TEST_SAVE = DATA_PATH / 'test_features.csv'


def combine(reactions, customers, transactions):
    reactions_customers = reactions.join(customers.set_index('customer_id'), on='customer_id')
    re_cu_st = reactions_customers.join(stories.set_index('story_id'), on='story_id')
    full_df = re_cu_st.join(transactions.set_index('customer_id'), on='customer_id')
    return full_df

if __name__ == '__main__':
    stories = pd.read_csv(STORIES_FILE, index_col=0)
    reactions_train = pd.read_csv(REACTION_TRAIN_FILE)
    reactions_test = pd.read_csv(REACTION_TEST_FILE)
    customers_train = pd.read_csv(CUSTOMER_TRAIN_FILE, index_col=0)
    customers_test = pd.read_csv(CUSTOMER_TEST_FILE, index_col=0)
    transactions = pd.read_csv(TRANSACTIONS_FILE, index_col=0)
    
    train_df = combine(reactions_train, customers_train, transactions)
    test_df = combine(reactions_test, customers_test, transactions)

    train_df.to_csv(TRAIN_SAVE, index=False)
    test_df.to_csv(TEST_SAVE, index=False)    
