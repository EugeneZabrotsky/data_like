import fasttext
import numpy as np
from nltk import tokenize
import pandas as pd
import config

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering


DATA_PATH = config.get_data_path()
STORIES_FILE = DATA_PATH / 'stories_processed.csv'


stories = pd.read_csv(STORIES_FILE, index_col=0,
                      converters={'all_text': eval, 'font_size': eval, 'guids': eval})


ft_model_path = DATA_PATH / '../fasttext' / 'cc.ru.300.bin'

model = fasttext.load_model(str(ft_model_path))
tokenizer = tokenize.TweetTokenizer()


vectors = []
for text in stories['all_text']:
    text = ' '.join(text).lower()
    text = tokenizer.tokenize(text)
    vector = np.array([model[word] for word in text]).mean(axis=0)
    vectors.append(vector)

cluster_model = AgglomerativeClustering(n_clusters=8)
clusters = cluster_model.fit_predict(vectors)
stories['clusters'] = clusters
stories.to_csv(STORIES_FILE)