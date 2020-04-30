import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

np.random.seed(42)


def load_meta_data(path='./data/movies_metadata.csv', filter_year=None):
    df = pd.read_csv(path)
    print('---- loaded movies meta data ----')
    df.id = pd.to_numeric(df.id, errors='coerce')
    df = df[df.id.notna()]
    df = df.drop_duplicates('id')
    df.id = df.id.astype('int64')
    df = df.set_index('id')

    if filter_year:
        print('total movies in table: {}'.format(len(df)))
        df = df[df.release_date >= str(filter_year)]
        print('removing movies before {}. movies after filter: {}'.format(filter_year, len(df)))

    df.genres = df.genres.apply(lambda x: [i['name'] for i in eval(x)])
    df = df[df.genres.apply(lambda x: len(x) > 0)]
    print('removing movies without genre label. movies after filter: {}'.format(len(df)))
    df['main_genre'] = df.genres.apply(lambda x: x[0])
    df['multiple_genres'] = df.genres.apply(lambda x: len(x) > 1)
    num_mult = sum(df.multiple_genres)
    print('movies with one genre: {}. movies with more then one genre: {}'.format(len(df) - num_mult, num_mult))

    fig = df.main_genre.value_counts().plot.bar()
    plt.title('Distribution of main genres of movies in data set')
    plt.xlabel('main genre')
    plt.ylabel('frequency')
    plt.savefig('./figures/full_genres_distribution.jpg', bbox_inches="tight")
    print('(saved figure: full_genres_distribution.jpg)')
    print('')
    return df


def print_stats(scores, save_fig=False, fig_title='', fig_name='', y_log=False):
    print('total number of ratings: {}'.format(len(scores)))
    print('number of different movies: {}'.format(scores.movieId.nunique()))
    print('number of different users: {}'.format(scores.userId.nunique()))
    print('data sparsity: {:.4f}'.format(len(scores) / (scores.movieId.nunique() * scores.userId.nunique())))
    print('average ratings per user: {}'.format(np.mean(scores.groupby('userId').movieId.count())))
    print('max ratings per user: {}'.format(np.max(scores.groupby('userId').movieId.count())))
    print('min ratings per user: {}'.format(np.min(scores.groupby('userId').movieId.count())))
    print('average ratings per movie: {}'.format(np.mean(scores.groupby('movieId').userId.count())))
    if save_fig:
        plt.figure()
        plt.hist(scores.groupby('userId').movieId.count())
        plt.title(fig_title)
        plt.xlabel('number of ratings')
        if y_log:
            plt.ylabel('log frequency')
            plt.yscale('log')
        else:
            plt.ylabel('frequency')
        plt.savefig('./figures/{}.jpg'.format(fig_name), bbox_inches="tight")
        print('(saved figure: {}.jpg)'.format(fig_name))
    print('')


def load_ratings(ratings_path='./data/ratings.csv', links_path='./data/links.csv',
                 meta_data_movies=None):
    scores = pd.read_csv(ratings_path)
    links = pd.read_csv(links_path)
    links = links[links.tmdbId.notna() & links.movieId.notna()]
    links = links.set_index('movieId', verify_integrity=True)['tmdbId'].astype('int64')
    scores['movieId'] = scores.movieId.map(links)
    scores = scores[scores.movieId.notna()]
    scores['movieId'] = scores.movieId.astype('int64')
    print('---- loaded ratings data ----')
    if meta_data_movies is not None:
        scores = scores[scores.movieId.isin(meta_data_movies)]
    print('-full ratings data set stats: -')
    print_stats(scores, save_fig=True, fig_title='Distribution of number of ratings per user',
                fig_name='original_user_ratings_distribution', y_log=True)

    n_movies, n_users = scores.movieId.nunique(), scores.userId.nunique()
    movies_per_user = scores.groupby('userId').movieId.count()
    users_ids = movies_per_user.sort_values().index[:int(0.98 * n_users)]
    scores = scores[scores.userId.isin(users_ids)]

    print('- starting iterative filtering of the ratings -')

    scores_ls = scores.copy()
    for i in range(12):
        n_movies, n_users = scores_ls.movieId.nunique(), scores_ls.userId.nunique()
        movies_per_user = scores_ls.groupby('userId').movieId.count()
        users_ids = movies_per_user.sort_values().index[int(-0.65 * n_users):]
        scores_ls = scores_ls[scores_ls.userId.isin(users_ids)]

        n_movies, n_users = scores_ls.movieId.nunique(), scores_ls.userId.nunique()
        users_per_movie = scores_ls.groupby('movieId').userId.count()
        mov_ids = users_per_movie.sort_values().index[int(-0.85 * n_movies):]
        scores_ls = scores_ls[scores_ls.movieId.isin(mov_ids)]

    scores_hs = scores_ls.copy()
    scores_hs = scores_hs.sample(frac=0.1)
    scores_ls = scores_ls[scores_ls.movieId.isin(scores_hs.movieId)]

    print('- Low-Sparsity ratings data set stats: -')
    print_stats(scores_ls, save_fig=True, fig_title='Distribution of number of ratings per user after filtering',
                fig_name='user_ratings_distribution_after_filtering')
    print('- High-Sparsity ratings data set stats: -')
    print_stats(scores_hs, save_fig=True, fig_title='Distribution of number of ratings per user after filtering',
                fig_name='user_ratings_distribution_after_filtering')

    return scores_hs, scores_ls


def df_to_sparse_matrix(df):
    features = pd.api.types.CategoricalDtype(sorted(df.feature.unique()), ordered=True)
    movies = pd.api.types.CategoricalDtype(sorted(df.movieId.unique()), ordered=True)

    row = df.movieId.astype(movies).cat.codes
    col = df.feature.astype(features).cat.codes

    mat = sparse.csr_matrix((df['value'], (row, col)),
                            shape=(movies.categories.size, features.categories.size))

    return mat, movies, features
