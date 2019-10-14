#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import numpy as np
import pandas as pd
import config

DATA_PATH = config.get_data_path()

transactions_file = DATA_PATH / 'transactions.csv'
transactions = pd.read_csv(transactions_file)

reaction_file = DATA_PATH / 'stories_reaction_train.csv'
reaction = pd.read_csv(reaction_file)

mcc_file = DATA_PATH / 'MCCs.csv'
mccs = pd.read_csv(mcc_file)

mccs.fillna('unknown', inplace=True)
mcc_map = dict(mccs[['MCC', 'Группа']].values)

transactions.head()
transactions['MCC_CODE_GROUP'] = transactions['merchant_mcc'].map(mcc_map).fillna("unknown")
unique_mcc = transactions['MCC_CODE_GROUP'].unique()

mcc_name_to_id = dict(zip(unique_mcc, range(len(unique_mcc))))

transactions['mcc_id'] = transactions['MCC_CODE_GROUP'].map(mcc_name_to_id)

mcc_groups = list(mcc_name_to_id.keys())

mcc_groups_amount = []
for group in mcc_groups:
    group_amount = 'AMOUNT_IF_' + group
    mcc_groups_amount.append(group_amount)
    transactions[group_amount] = (transactions['MCC_CODE_GROUP'] == group) * transactions['transaction_amt']

cl = pd.DataFrame({"customer_id": np.unique(transactions["customer_id"])})
grouped_cl = transactions.groupby(['customer_id'])
def add_feature(feature_name, col, func, res = cl, grouped = grouped_cl):
    res[feature_name] = grouped.agg({col: func})[col].values

add_feature('sum_amount', 'transaction_amt', np.sum)
add_feature('mean_amount', 'transaction_amt', np.mean)
add_feature('num_trans', 'transaction_amt', len)
add_feature('std_amount', 'transaction_amt', np.std)

cl['std_amount_normalized'] = cl['std_amount'] / cl['mean_amount']
for group in mcc_groups_amount:
    add_feature('sum_' + group, group, np.sum)
    cl['sum_' + group + '_percentage'] = cl['sum_' + group] / cl['sum_amount']
cl.fillna(0, inplace = True)
CUSTOMER_FEATURES_FILE = DATA_PATH / 'customer_transactions.csv'

cl.to_csv(CUSTOMER_FEATURES_FILE)