import numpy as np
import pandas as pd
import dostoevsky
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from collections import defaultdict

import config


DATA_PATH = config.get_data_path()

STORIES_FILE = DATA_PATH / 'stories_processed.csv'
REACTION_TRAIN_FILE = DATA_PATH / 'stories_reaction_train.csv'
REACTION_TEST_FILE = DATA_PATH / 'stories_reaction_test.csv'

CUSTOMER_TRAIN_FILE = DATA_PATH / 'customer_train_processed.csv'
CUSTOMER_TEST_FILE = DATA_PATH / 'customer_test_processed.csv'

TRANSACTIONS_FILE = DATA_PATH / 'customer_transactions.csv'

STORIES_SAVE = DATA_PATH / 'stories_features.csv'
REACTOINS_TRAIN_SAVE = DATA_PATH / 'reactions_features_train.csv'
REACTOINS_TEST_SAVE = DATA_PATH / 'reactions_features_test.csv'

CUSTOMER_TRAIN_SAVE = DATA_PATH / 'customer_features_train.csv'
CUSTOMER_TEST_SAVE = DATA_PATH / 'customer_features_test.csv'

TRANSACTIONS_SAVE = DATA_PATH / 'transactions_features.csv'


def process_customers(customers):
    product_cols = [f'product_{i}' for i in range(7)]
    
    cols_to_drop = ['job_title', 'first_session_dttm']
    customers = customers.fillna(-1)
    customers = customers.drop(cols_to_drop, axis=1)
    return customers

def get_story_features(story):
    text = story['all_text']
    
    features = dict()
    features['num_messages'] = len(text)
    features['text_len'] = len(' '.join(text))
    features['num_guids'] = len(story['guids'])
    return features

def df_from_feature_dicts(rows):
    feature_names = rows[0].keys()
    future_df = defaultdict(list)
    
    for name in feature_names:
        for row in rows:
            future_df[name].append(row[name])
    
    return pd.DataFrame(future_df)

def stories_features(stories):
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    text_info = model.predict(stories['all_text'].apply(' '.join))    
    text_featrues_df = df_from_feature_dicts(text_info)
    
    feature_list = []
    for i, story in stories.iterrows():
        feature_list.append(get_story_features(story))
        
    story_features = df_from_feature_dicts(feature_list)
    
    stories_full = pd.concat([stories, text_featrues_df, story_features], axis=1)
    
    cols_to_drop = ['all_text', 'urls', 'font_sizes', 'guids']
    stories_full = stories_full.drop(cols_to_drop, axis=1)
    return stories_full

def process_reactions(reactions):
    reactions = reactions.copy()
    weekday = reactions['event_dttm'].dt.weekday
    weekday.name = 'weekday'
    day = reactions['event_dttm'].dt.day
    day.name = 'day'
    hour = reactions['event_dttm'].dt.hour
    hour.name = 'hour'
    minute = reactions['event_dttm'].dt.minute
    minute.name = 'minute'
    is_weekend = weekday.between(5, 6).astype(int)
    is_weekend.name = 'is_weeked'
        
    if 'event' in reactions.columns:
        label2id = config.label2id()
        reactions['event'] = reactions['event'].map(label2id)
        
    new_reactions = pd.concat([reactions, weekday, day, hour, minute, is_weekend], axis=1)
#     cols_to_drop = ['event_dttm']
#     new_reactions = new_reactions.drop(cols_to_drop, axis=1)
    
    return new_reactions

def process_transactions(transactions):
    return transactions

if __name__ == '__main__':     
    stories = pd.read_csv(STORIES_FILE, index_col=0, converters={'all_text': eval, 'font_size': eval, 'guids': eval})
    reactions_train = pd.read_csv(REACTION_TRAIN_FILE, parse_dates=['event_dttm'])
    reactions_test = pd.read_csv(REACTION_TEST_FILE, parse_dates=['event_dttm'])
    customers_train = pd.read_csv(CUSTOMER_TRAIN_FILE, index_col=0)
    customers_test = pd.read_csv(CUSTOMER_TEST_FILE, index_col=0)
    transactions = pd.read_csv(TRANSACTIONS_FILE, index_col=0)

    new_stories = stories_features(stories)
    new_customers_train = process_customers(customers_train)
    new_customers_test = process_customers(customers_test)
    new_transactions = process_transactions(transactions)
    new_reactions_train = process_reactions(reactions_train)
    new_reactions_test = process_reactions(reactions_test)    
    
    new_stories.to_csv(STORIES_SAVE)
    new_customers_train.to_csv(CUSTOMER_TRAIN_SAVE)
    new_customers_test.to_csv(CUSTOMER_TEST_SAVE)    
    new_transactions.to_csv(TRANSACTIONS_SAVE)
    new_reactions_train.to_csv(REACTOINS_TRAIN_SAVE)
    new_reactions_test.to_csv(REACTOINS_TEST_SAVE)