import numpy as np
import pandas as pd
import data_utils
import pickle
import NMF
from scipy.stats import ttest_1samp
from sklearn.metrics import normalized_mutual_info_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from collections import Counter

warnings.simplefilter(action='ignore', category=FutureWarning)


def calc_score_significance(true, pred, score_function):
    my_score = score_function(true, pred)
    shuffled_true = true.copy()
    a = []
    for i in range(1000):
        np.random.shuffle(shuffled_true)
        a.append(score_function(shuffled_true, pred))
    print('score: ', my_score,' random score mean: ', np.mean(a),
          ' randon score std: ', np.std(a))
    print(ttest_1samp(a, my_score))


def semi_supervised_classification(true, pred, train_frac=0.3):
    all_trues = []
    all_preds = []
    for c in set(pred):
        mask = (pred==c)
        cur_true = true[mask]
        true_train, true_test = train_test_split(cur_true, train_size=0.3)
        c = Counter(true_train)
        l = c.most_common(1)[0][0]
        all_trues += list(true_test)
        all_preds += len(true_test) * [l]
    return f1_score(all_trues, all_preds, average='weighted'), accuracy_score(all_trues, all_preds)


md_df = data_utils.load_meta_data(filter_year=1990)

try:
    with open('./data/ratings_filtered.pkl', 'rb') as f:
        ratings_df = pickle.load(f)
except IOError:
    ratings_df = data_utils.load_ratings(meta_data_movies=md_df.index)
    with open('./data/ratings_filtered.pkl', 'wb') as f:
        pickle.dump(ratings_df, f)
        print('(saved ratings_filtered.pkl)')
try:
    with open('./data/credits_filtered.pkl', 'rb') as f:
        credits_df = pickle.load(f)
except IOError:
    credits_df = data_utils.load_credits(meta_data_movies=md_df.index)
    with open('./data/credits_filtered.pkl', 'wb') as f:
        pickle.dump(credits_df, f)
        print('(saved credits_filtered.pkl)')

print('Number of movies in ratings table: ', ratings_df.movieId.nunique())
print('Number of movies in credits table: ', credits_df.movieId.nunique())
common_ind = set(credits_df.movieId) & set(ratings_df.movieId)
print('Number of movies in common: ', len(common_ind))
print('')

ratings_df = ratings_df[ratings_df.movieId.isin(common_ind)]
credits_df = credits_df[credits_df.movieId.isin(common_ind)]


n_trails = 50
n_components = 35

ratings_df.rename(columns={'userId': 'feature', 'rating': 'value'}, inplace=True)
mat, movies, users = data_utils.df_to_sparse_matrix(ratings_df)
nmf = NMF.NMF(n_components=n_components)
W, H = nmf.decompose(mat)

clusters = W.argmax(axis=1)
md_df = md_df.loc[movies.categories.values]
labels = md_df.main_genre.values
high_conf = ~md_df.multiple_genres
hconf_labels = labels[high_conf]
hconf_clusters = clusters[high_conf]

print('test MIS on all labels')
calc_score_significance(labels, clusters, normalized_mutual_info_score)
print('')


print('test MIS on high confidence labels')
calc_score_significance(hconf_labels, hconf_clusters, normalized_mutual_info_score)
print('')

print('test semi-supervised f1-score on all labels')
take_f1_score = lambda x, y: semi_supervised_classification(x, y)[0]
calc_score_significance(labels, clusters, take_f1_score)
print('')

print('test semi-supervised accuracy on all labels')
take_accuracy = lambda x, y: semi_supervised_classification(x, y)[1]
calc_score_significance(labels, clusters, take_accuracy)
print('')

