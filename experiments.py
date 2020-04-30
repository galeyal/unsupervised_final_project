import numpy as np
import pandas as pd
import data_utils
import NMF
from scipy.stats import ttest_1samp
from sklearn.metrics import normalized_mutual_info_score, f1_score, accuracy_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import train_test_split
import warnings
from itertools import product
from tqdm import tqdm
from functools import partial
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

weighted_f1_score = partial(f1_score, average='weighted')

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def calc_score_significance(true, pred, score_function):
    my_score = score_function(true, pred)
    shuffled_true = true.copy()
    a = []
    for i in range(500):
        np.random.shuffle(shuffled_true)
        a.append(score_function(shuffled_true, pred))
    t, pval = ttest_1samp(a, my_score)
    return my_score, t, pval


def semi_supervised_classify(true, cluster, train_frac=0.5):
    true_train, true_test, cluster_train, cluster_test = train_test_split(true, cluster, train_size=train_frac)
    train = pd.DataFrame({'label': true_train, 'cluster': cluster_train})
    tags = train.groupby('cluster').agg(lambda x: x.value_counts().index[0])
    test = pd.DataFrame({'label': true_test, 'cluster': cluster_test})
    test['pred'] = test['cluster'].map(tags.label).fillna(train['label'].value_counts().index[0])

    f1, f1_t, f1_p = calc_score_significance(test['label'].values, test['pred'].values, weighted_f1_score)
    acc, acc_t, acc_p = calc_score_significance(test['label'].values, test['pred'].values, accuracy_score)
    return f1, f1_t, f1_p, acc, acc_t, acc_p


def semi_supervised_classify_random(true, train_frac=0.5):
    true_train, true_test = train_test_split(true, train_size=train_frac)
    test_pred = np.random.choice(true_train, true_test.shape)
    return f1_score(true_test, test_pred, average='weighted'), accuracy_score(true_test, test_pred)


def run_single_nmf_experiment(mat, n_components, apply_nan, labels):
    if apply_nan:
        nmf = NMF.NMF(n_components=n_components, apply_nan_mask=True, nan_weight=0.1)
    else:
        nmf = NMF.NMF(n_components=n_components)

    W, H = nmf.decompose(mat)
    clusters = W.argmax(axis=1)

    nmi, nmi_t, nmi_p = calc_score_significance(labels, clusters, normalized_mutual_info_score)
    if nmi_p >= 0.05:
        print('non conclusive NMI with {} and nan_mask={} '.format(n_components, apply_nan),
              nmi_t, nmi_p)

    sill = silhouette_score(mat.toarray(), clusters)

    accs = []
    f1s = []
    for i in range(20):
        f1, f1_t, f1_p, acc, acc_t, acc_p = semi_supervised_classify(labels, clusters)
        accs.append(acc)
        f1s.append(f1)
        if f1_p >= 0.05:
            print('non conclusive f1 score with {} and nan_mask={} '.format(n_components, apply_nan),
                  f1_t, f1_p)
        if acc_p >= 0.05:
            print('non conclusive accuracy score with {} and nan_mask={} '.format(n_components, apply_nan),
                  acc_t, acc_p)

    return {'nmi': nmi, 'nmi_p': nmi_p, 'sill': sill,
            'acc': np.mean(accs), 'acc_std': np.std(accs),
            'f1': np.mean(f1s), 'f1_std': np.std(f1s)}


def run_all_nmf_experiments(md_df, ls_ratings_df, hs_ratings_df):
    res_df = pd.DataFrame(columns=['set', 'n', 'nan', 'nmi', 'nmi_p', 'sill',
                                   'acc', 'acc_std', 'f1', 'f1_std'])

    mat, movies, users = data_utils.df_to_sparse_matrix(ls_ratings_df)
    md_df = md_df.loc[movies.categories.values]
    labels = md_df.main_genre.values

    print(' -- running NMF experiment on Low-Sparsity --')
    for n, nan in tqdm(list(product(range(5, 55, 5), [True, False]))):
        r = {'set': 'LS', 'n': n, 'nan': nan}
        r.update(run_single_nmf_experiment(mat, n_components=n, apply_nan=nan, labels=labels))
        res_df = res_df.append(r, ignore_index=True)
        res_df.to_pickle('./results/all_NMF_results.pkl')

    mat, movies, users = data_utils.df_to_sparse_matrix(hs_ratings_df)
    md_df = md_df.loc[movies.categories.values]
    labels = md_df.main_genre.values

    print('-- running NMF experiment on High-Sparsity --')
    for n, nan in tqdm(list(product(range(5, 55, 5), [True, False]))):
        r = {'set': 'HS', 'n': n, 'nan': nan}
        r.update(run_single_nmf_experiment(mat, n_components=n, apply_nan=nan, labels=labels))
        res_df = res_df.append(r, ignore_index=True)
        res_df.to_pickle('./results/all_NMF_results.pkl')
    return res_df


def run_knn_experiment(mat, n_components, labels):
    kmeans = KMeans(n_components)
    clusters = kmeans.fit_predict(mat)

    nmi, nmi_t, nmi_p = calc_score_significance(labels, clusters, normalized_mutual_info_score)
    if nmi_p >= 0.05:
        print('non conclusive NMI with {}'.format(n_components),
              nmi_t, nmi_p)

    sill = silhouette_score(mat, clusters)

    accs = []
    f1s = []
    for i in range(20):
        f1, f1_t, f1_p, acc, acc_t, acc_p = semi_supervised_classify(labels, clusters)
        accs.append(acc)
        f1s.append(f1)
        if f1_p >= 0.05:
            print('non conclusive f1 score with {} '.format(n_components),
                  f1_t, f1_p)
        if acc_p >= 0.05:
            print('non conclusive accuracy score with {}'.format(n_components),
                  acc_t, acc_p)

    return {'nmi': nmi, 'nmi_p': nmi_p, 'sill': sill,
            'acc': np.mean(accs), 'acc_std': np.std(accs),
            'f1': np.mean(f1s), 'f1_std': np.std(f1s)}


def run_all_mds_experiments(md_df, ls_ratings_df, hs_ratings_df):
    res_df = pd.DataFrame(columns=['set', 'n_embed', 'n_cluster', 'nmi', 'nmi_p', 'sill',
                                   'acc', 'acc_std', 'f1', 'f1_std'])

    print('-- running MDS experiment on Low-Sparsity --')
    mat, movies, users = data_utils.df_to_sparse_matrix(ls_ratings_df)
    md_df = md_df.loc[movies.categories.values]
    labels = md_df.main_genre.values
    cosine_diss = mat.toarray()
    np.fill_diagonal(cosine_diss, 1)
    cosine_diss = np.sqrt(cosine_distances(cosine_diss))
    np.fill_diagonal(cosine_diss, 0)

    for n_embed in range(5, 25, 5):
        mds = MDS(n_embed, metric=True, n_jobs=2, n_init=2, dissimilarity='precomputed', max_iter=1000)
        print('-- fitting MDS {} components --'.format(n_embed))
        new_rep = mds.fit_transform(cosine_diss)
        print('running KMeans Experiment')
        for nc in tqdm(range(5, 55, 5)):
            r = {'set': 'LS', 'n_embed': n_embed, 'n_cluster': nc}
            r.update(run_knn_experiment(new_rep, n_components=nc, labels=labels))
            res_df = res_df.append(r, ignore_index=True)
            res_df.to_pickle('./results/all_MDS_results.pkl')

    print('-- running MDS experiment on High-Sparsity --')
    mat, movies, users = data_utils.df_to_sparse_matrix(hs_ratings_df)
    md_df = md_df.loc[movies.categories.values]
    labels = md_df.main_genre.values
    cosine_diss = mat.toarray()
    np.fill_diagonal(cosine_diss, 1)
    cosine_diss = np.sqrt(cosine_distances(cosine_diss))
    np.fill_diagonal(cosine_diss, 0)

    for n_embed in range(5, 25, 5):
        mds = MDS(n_embed, metric=True, n_jobs=2, n_init=2, dissimilarity='precomputed', max_iter=1000)
        print('-- fitting MDS {} components --'.format(n_embed))
        new_rep = mds.fit_transform(cosine_diss)
        print('running KMeans Experiment')
        for nc in tqdm(range(5, 55, 5)):
            r = {'set': 'HS', 'n_embed': n_embed, 'n_cluster': nc}
            r.update(run_knn_experiment(new_rep, n_components=nc, labels=labels))
            res_df = res_df.append(r, ignore_index=True)
            res_df.to_pickle('./results/all_MDS_results.pkl')
    return res_df
