import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
np.random.seed(42)

def load_meta_data(path='./data/movies_metadata.csv', filter_year=None):
    df = pd.read_csv(path)
    print ('---- loaded movies meta data ----')
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
    plt.savefig('./figures/full_genres_distribution.jpg')
    print('saved figure: full_genres_distribution.jpg')
    print('')
    return df


def load_ratings(ratings_path='./data/ratings.csv', links_path='./data/links.csv',
                 meta_data_movies=None):
    scores = pd.read_csv(ratings_path)
    links = pd.read_csv(links_path)
    links = links[links.tmdbId.notna() & links.movieId.notna()]
    links = links.set_index('movieId', verify_integrity=True)['tmdbId'].astype('int64')
    scores['movieId'] = scores.movieId.map(links)
    scores = scores[scores.movieId.notna()]
    scores['movieId'] = scores.movieId.astype('int64')
    print('---- loaded scores data ----')
    if meta_data_movies is not None:
        scores = scores[scores.movieId.isin(meta_data_movies)]

    print('ratings data stats:')
    print('total number of ratings: {}'.format(len(scores)))
    print('number of different movies: {}'.format(scores.movieId.nunique()))
    print('number of different users: {}'.format(scores.userId.nunique()))
    print('data sparsity: {:.4f}'.format(len(scores) / (scores.movieId.nunique() * scores.userId.nunique())))
    print('average ratings per user: {}'.format(np.mean(scores.groupby('userId').movieId.count())))
    print('max ratings per user: {}'.format(np.max(scores.groupby('userId').movieId.count())))
    print('min ratings per user: {}'.format(np.min(scores.groupby('userId').movieId.count())))
    print('average ratings per movie: {}'.format(np.mean(scores.groupby('movieId').userId.count())))

    plt.figure()
    plt.hist(scores.groupby('userId').movieId.count())
    plt.title('Distribution of number of ratings per user')
    plt.xlabel('number of ratings')
    plt.ylabel('log frequency')
    plt.yscale('log')
    plt.savefig('./figures/original_user_ratings_distribution.jpg')
    print('saved figure: original_user_ratings_distribution.jpg')

    print('starting iterative filtering of the ratings')
    n_movies, n_users = scores.movieId.nunique(), scores.userId.nunique()
    movies_per_user = scores.groupby('userId').movieId.count()
    users_ids = movies_per_user.sort_values().index[:int(0.96 * n_users)]
    scores = scores[scores.userId.isin(users_ids)]

    for i in range(9):
        n_movies, n_users = scores.movieId.nunique(), scores.userId.nunique()
        movies_per_user = scores.groupby('userId').movieId.count()
        users_ids = movies_per_user.sort_values().index[int(-0.6 * n_users):]
        scores = scores[scores.userId.isin(users_ids)]

        users_per_movie = scores.groupby('movieId').userId.count()
        mov_ids = users_per_movie.sort_values().index[int(-0.85 * n_movies):]
        scores = scores[scores.movieId.isin(mov_ids)]
    print('')
    print('new statistics after filtering:')
    print('total number of ratings: {}'.format(len(scores)))
    print('number of different movies: {}'.format(scores.movieId.nunique()))
    print('number of different users: {}'.format(scores.userId.nunique()))
    print('data sparsity: {:.4f}'.format(len(scores) / (scores.movieId.nunique() * scores.userId.nunique())))
    print('average ratings per user: {}'.format(np.mean(scores.groupby('userId').movieId.count())))
    print('max ratings per user: {}'.format(np.max(scores.groupby('userId').movieId.count())))
    print('min ratings per user: {}'.format(np.min(scores.groupby('userId').movieId.count())))
    print('average ratings per movie: {}'.format(np.mean(scores.groupby('movieId').userId.count())))

    plt.figure()
    plt.hist(scores.groupby('userId').movieId.count())
    plt.title('Distribution of number of ratings per user after filtering')
    plt.xlabel('number of ratings')
    plt.ylabel('frequency')
    plt.savefig('./figures/user_ratings_distribution_after_filter.jpg')
    print('saved figure: user_ratings_distribution_after_filter.jpg')
    print('')
    return scores


def load_credits(credits_path='./data/credits.csv', meta_data_movies=None):
    credits = pd.read_csv(credits_path)
    credits = credits.drop_duplicates('id')
    credits.rename(columns={'id': 'movieId'}, inplace=True)
    credits.cast = credits.cast.apply(lambda x: set([i['name'] for i in eval(x)]))
    credits.crew = credits.crew.apply(lambda x: set([i['name'] for i in eval(x)]))
    credits.crew = credits.apply(lambda x: x.cast | x.crew, axis=1)
    credits = pd.DataFrame({'movieId': np.repeat(credits.movieId.values, credits.crew.map(len)),
                            'crew': np.concatenate(credits.crew.map(list).values)})

    print('---- loaded credits data ----')
    if meta_data_movies is not None:
        credits = credits[credits.movieId.isin(meta_data_movies)]

    print('credits data stats:')
    print('total number of credits: {}'.format(len(credits)))
    print('number of different movies: {}'.format(credits.movieId.nunique()))
    print('number of different crew members: {}'.format(credits.crew.nunique()))
    print('data sparsity: {:.4f}'.format(len(credits) / (credits.movieId.nunique() * credits.crew.nunique())))
    print('average movies per crew member: {}'.format(np.mean(credits.groupby('crew').movieId.count())))
    print('average crew members per movie: {}'.format(np.mean(credits.groupby('movieId').crew.count())))

    plt.figure()
    plt.hist(credits.groupby('crew').movieId.count())
    plt.title('Distribution of number of movies per crew member')
    plt.xlabel('number of movies')
    plt.ylabel('log frequency')
    plt.yscale('log')
    plt.savefig('./figures/original_movies_per_crew_distribution.jpg')
    print('saved figure: original_moveis_per_crew_distribution.jpg')

    print('starting iterative filtering of the credits table')

    for i in range(9):
        n_movies, n_crews = credits.movieId.nunique(), credits.crew.nunique()
        movies_per_crew = credits.groupby('crew').movieId.count()
        crews_ids = movies_per_crew.sort_values().index[int(-0.6 * n_crews):]
        credits = credits[credits.crew.isin(crews_ids)]

        crews_per_movie = credits.groupby('movieId').crew.count()
        mov_ids = crews_per_movie.sort_values().index[int(-0.85 * n_movies):]
        credits = credits[credits.movieId.isin(mov_ids)]

    print('')
    print('new statistics after filtering:')
    print('total number of credits: {}'.format(len(credits)))
    print('number of different movies: {}'.format(credits.movieId.nunique()))
    print('number of different crew members: {}'.format(credits.crew.nunique()))
    print('data sparsity: {:.4f}'.format(len(credits) / (credits.movieId.nunique() * credits.crew.nunique())))
    print('average movies per crew member: {}'.format(np.mean(credits.groupby('crew').movieId.count())))
    print('average crew members per movie: {}'.format(np.mean(credits.groupby('movieId').crew.count())))

    plt.figure()
    plt.hist(credits.groupby('crew').movieId.count())
    plt.title('Distribution of number of movies per crew member')
    plt.xlabel('number of movies')
    plt.ylabel('frequency')
    plt.savefig('./figures/movies_per_crew_distribution_after_filter.jpg')
    print('saved figure: movies_per_crew_distribution_after_filter.jpg')
    print('')
    return credits


def df_to_sparse_matrix(df):
    features = pd.api.types.CategoricalDtype(sorted(df.feature.unique()), ordered=True)
    movies = pd.api.types.CategoricalDtype(sorted(df.movieId.unique()), ordered=True)

    row = df.movieId.astype(movies).cat.codes
    col = df.feature.astype(features).cat.codes

    mat = sparse.csr_matrix((df['value'], (row, col)),
                            shape=(movies.categories.size, features.categories.size))

    return mat, movies, features


def train_test_split_to_sparse_matrix(df, movies, features, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size)

    row = train_df.movieId.astype(movies).cat.codes
    col = train_df.feature.astype(features).cat.codes
    train_mat = sparse.csr_matrix((train_df['value'], (row, col)),
                                  shape=(movies.categories.size, features.categories.size))

    row = test_df.movieId.astype(movies).cat.codes
    col = test_df.feature.astype(features).cat.codes
    test_mat = sparse.csr_matrix((test_df['value'], (row, col)),
                                  shape=(movies.categories.size, features.categories.size))

    return train_mat, test_mat

