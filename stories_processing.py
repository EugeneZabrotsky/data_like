#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import pandas as pd
import json
from unicodedata import normalize
import re
from collections import defaultdict

import config

DATA_PATH = config.get_data_path()

STORIES_FILE = DATA_PATH / 'stories_description.csv'
STORIES_SAVE = DATA_PATH / 'stories_processed.csv'


def get_text_amount(all_text, font_sizes):
    assert len(all_text) == len(font_sizes)
    lengths = np.array(list(map(len, all_text)))
    sizes = (np.array(font_sizes) / 100)**2
    return (lengths * sizes).sum()

def parse_story(story_raw, story_id):
    story = json.loads(story_raw)

    p = re.compile('"text":\s*"([^"]+)"')
    all_text = re.findall(p, story_raw)
    all_text = [normalize('NFKC', text) for text in all_text]

    p = re.compile('"text":\s*"[^"]+",\s*"font_size":\s*([^"]+),')
    font_sizes = re.findall(p, story_raw)
    
    p = re.compile('"(https?://.*?)"')
    urls = re.findall(p, story_raw)

    font_sizes_int = []
    for font_size in font_sizes:
        try:
            font_sizes_int.append(int(font_size))
        except ValueError:
            pass
    mean_font_size = np.mean(font_sizes_int)

    assert(len(font_sizes) == len(all_text))

    p = re.compile('"guid":\s*"([^"]+)"')
    guids = re.findall(p, story_raw)   

    features = dict()
    features['num_pages'] = len(story['content'])
    features['all_text'] = all_text
    features['urls'] = urls
    features['num_urls'] = len(urls)
    features['num_elements'] = len(guids)
    features['mean_font_size'] = mean_font_size
    features['font_sizes'] = font_sizes_int
    features['guids'] = guids
    features['text_amount'] = get_text_amount(all_text, font_sizes_int)
    features['story_id'] = story_id
    
    return features

def parse_stories(stories):
    features_list = []
    for i in range(len(stories)):
        features = parse_story(stories['story_json'][i], stories['story_id'][i])
        features_list.append(features)

    features_data = defaultdict(list)
    feature_names = features_list[0].keys()

    for features in features_list:
        for name in feature_names:
            features_data[name].append(features[name])

    stories_df = pd.DataFrame(features_data)
    return stories_df

if __name__ == '__main__':
    stories = pd.read_csv(STORIES_FILE)
    stories['story_json'] = stories['story_json'].apply(lambda x: x.replace('\\\\', '\\'))
    
    stories_df = parse_stories(stories)
    stories_df.to_csv(STORIES_SAVE)

