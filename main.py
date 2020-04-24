import numpy as np
import pandas as pd
import data_utils
import pickle
import NMF
from tqdm import tqdm


def run_NMF_experiment(df, n_components, test_size=0.2):
    rat_mat, movies, users = data_utils.df_to_sparse_matrix(df)
    train_mat, test_mat = data_utils.train_test_split_to_sparse_matrix(df, movies, users, test_size=test_size)
    nmf = NMF.NMF(n_components=n_components)
    W, H = nmf.decompose(train_mat)
    rec = W@H
    train_cor = train_mat.nonzero()
    test_cor = test_mat.nonzero()
    train_error = 0.5 * np.sum(np.square(rec[train_cor]-train_mat[train_cor])) / train_mat.count_nonzero()
    test_error = 0.5 * np.sum(np.square(rec[test_cor]-test_mat[test_cor])) / test_mat.count_nonzero()
    return {'n_components': n_components, 'train_error': train_error, 'test_error': test_error}

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

ratings_df.rename(columns={'userId': 'feature', 'rating': 'value'}, inplace=True)

res_df = pd.DataFrame(columns=['n_components', 'train_error', 'test_error'])
n_trails = 50
for nc in range(5, 55, 5):
    print('running NMF on ratings experiment ({} times) with n_components={}'.format(n_trails, nc))
    res_df = res_df.append([run_NMF_experiment(ratings_df, nc) for i in tqdm(range(n_trails))])
    res_df.to_csv('./results/ratings_experiments.csv')
