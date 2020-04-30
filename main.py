import numpy as np
import data_utils
import pickle
import NMF
import argparse
from experiments import run_all_nmf_experiments, run_all_mds_experiments, semi_supervised_classify_random
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt


def main():
    # parsing command line params
    parser = argparse.ArgumentParser(description='Loading the data and running all experiments')
    parser.add_argument('--metadata-file', action='store', help='location of movies metadata file',
                        default='./data/movies_metadata.csv')
    parser.add_argument('--ratings-file', action='store', help='location of ratings file',
                        default='./data/ratings.csv')
    parser.add_argument('--links-file', action='store', help='location of linksfile',
                        default='./data/links.csv')

    parser.set_defaults(dataset_creation=True, nmf_experiment=True, mds_experiment=True)

    parser.add_argument('--skip-dataset-creation', dest='dataset_creation', action='store_false',
                        help='flag to skip data-set creation')
    parser.add_argument('--skip-nmf-experiment', dest='nmf_experiment', action='store_false',
                        help='flag to skip nmf experiments')
    parser.add_argument('--skip-mds-experiment', dest='mds_experiment', action='store_false',
                        help='flag to skip mds experiments')
    args = parser.parse_args()

    # data set creation
    if args.dataset_creation:
        md_df = data_utils.load_meta_data(path=args.metadata_file, filter_year=1990)
        hs_ratings_df, ls_ratings_df = data_utils.load_ratings(ratings_path=args.ratings_file,
                                                               links_path=args.links_file, meta_data_movies=md_df.index)

        with open('./data/md_filtered.pkl', 'wb') as f:
            pickle.dump(md_df, f)
            print('(saved md_filtered.pkl)')

        with open('./data/ls_ratings_filtered.pkl', 'wb') as f:
            pickle.dump(ls_ratings_df, f)
            print('(saved ls_ratings_filtered.pkl)')

        with open('./data/hs_ratings_filtered.pkl', 'wb') as f:
            pickle.dump(hs_ratings_df, f)
            print('(saved hs_ratings_filtered.pkl)')

    else:
        print('-- skipping dataset creation and loading existing pickle files --\n')
        try:
            with open('./data/md_filtered.pkl', 'rb') as f:
                md_df = pickle.load(f)
            with open('./data/ls_ratings_filtered.pkl', 'rb') as f:
                ls_ratings_df = pickle.load(f)
            with open('./data/hs_ratings_filtered.pkl', 'rb') as f:
                hs_ratings_df = pickle.load(f)
        except IOError as e:
            print('dataset pickle not found try to run without the flag --skip-dataset-creation')
            raise e

        print('- Low-Sparsity ratings data set stats: -')
        data_utils.print_stats(ls_ratings_df)
        print('- High-Sparsity ratings data set stats: -')
        data_utils.print_stats(hs_ratings_df)

    ls_ratings_df.rename(columns={'userId': 'feature', 'rating': 'value'}, inplace=True)
    hs_ratings_df.rename(columns={'userId': 'feature', 'rating': 'value'}, inplace=True)

    # plotting label distribution
    movie_index = hs_ratings_df.movieId

    genre_map = {
        'Comedy': 'Comedy',
        'Drama': 'Drama',
        'Thriller': 'Action',
        'Action': 'Action',
        'Crime': 'Action',
        'Adventure': 'Action',
        'Horror': 'Horror'
    }

    md_df.main_genre = md_df.main_genre.map(genre_map).fillna('Other')
    fig = md_df.loc[movie_index].main_genre.value_counts().plot.bar()
    plt.title('Distribution of main genres of movies in experiments')
    plt.xlabel('main genre')
    plt.ylabel('frequency')
    plt.savefig('./figures/experiments_genres_distribution.jpg', bbox_inches="tight")
    print('(saved figure: experiments_genres_distribution.jpg)')
    print('')

    # running nmf experiments or loading results
    if args.nmf_experiment:
        print('---- running NMF experiments -----')
        nmf_res = run_all_nmf_experiments(md_df, ls_ratings_df, hs_ratings_df)

    else:
        print('-- skipping nmf examples loading existing results files --\n')
        try:
            with open('./results/all_NMF_results.pkl', 'rb') as f:
                nmf_res = pickle.load(f)
        except IOError as e:
            print('results pickle not found try to run without the flag --skip-nmf-experiment')
            raise e
    nmf_res.nan = nmf_res.nan.map(bool)

    if args.mds_experiment:
        print('---- running MDS experiments -----')
        mds_res = run_all_mds_experiments(md_df, ls_ratings_df, hs_ratings_df)

    # running mds experiments or loading results
    else:
        print('-- skipping mds examples loading existing results files --\n')
        try:
            with open('./results/all_MDS_results.pkl', 'rb') as f:
                mds_res = pickle.load(f)
        except IOError as e:
            print('results pickle not found try to run without the flag --skip-mds-experiment')
            raise e
    # Running random experiment
    labels = md_df.loc[movie_index].main_genre.values
    f1s = []
    accs = []
    print('-- running random experiment --')
    for i in range(100):
        f, a = semi_supervised_classify_random(labels)
        f1s.append(f)
        accs.append(a)
    rand = {'rand_f1': np.mean(f1s), 'rand_f1_err': np.std(f1s), 'rand_acc': np.mean(accs),
            'rand_acc_err': np.std(accs)}
    print('')

    # ---extracting all results plots---

    # creating a two dimensinal plot of the data
    print('-- computing 2-dimensional embedding using MDS (might take couple minutes)--')
    mat, movies, users = data_utils.df_to_sparse_matrix(ls_ratings_df)
    md_df = md_df.loc[movies.categories.values]
    labels = md_df.main_genre.values
    mds = MDS(2, metric=True, n_init=1, dissimilarity='precomputed', max_iter=500)
    dmat = mat.toarray()
    cosine_diss = dmat
    np.fill_diagonal(cosine_diss, 1)
    cosine_diss = np.sqrt(cosine_distances(cosine_diss))
    np.fill_diagonal(cosine_diss, 0)
    rep_2d = mds.fit_transform(cosine_diss)

    inds = np.array(range(len(labels)))
    np.random.shuffle(inds)
    labels_s = labels[inds[:300]]
    rep_2d_s = rep_2d[inds[:300], :]
    plt.figure()
    for l in np.unique(labels_s):
        mask = (labels_s == l)
        plt.scatter(rep_2d_s[mask, 0], rep_2d_s[mask, 1], label=l)
    plt.title('300 movies from the data set after MDS projection to 2D')
    plt.xlabel('first coordinate')
    plt.ylabel('second coordinate')
    plt.legend(title='genre')
    plt.savefig('./figures/2d_projection_of_data.jpg')
    print('(saved figure: 2d_projection_of_data.jpg)')
    print('')

    print('--- saving plots for all NMF results ---')
    # plot all nmf nmi results
    plt.figure()
    plt.title('Normalized-Mutual-Information as function of NMF dimensions for different cases')
    plt.ylabel('NMI score')
    plt.xlabel('NMF embedding size')
    plt.plot(nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].n, nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].nmi,
             label='Low-Sparsity | Weighted NMF', color='r')
    plt.plot(nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].n, nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].nmi,
             label='Low-Sparsity | NMF',
             color='b')
    plt.plot(nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].n, nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].nmi,
             label='High-Sparsity | Weighted NMF',
             color='r', linestyle='--')
    plt.plot(nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].n, nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].nmi,
             label='High-Sparsity | NMF',
             color='b', linestyle='--')
    plt.legend(loc='upper left', title='case', prop={'size': 9})
    plt.savefig('./figures/nmf_nmi_results.jpeg', bbox_inches="tight")
    print('(saved figure: nmf_nmi_results.jpeg)')

    # plot all nmf f1 results
    plt.figure()
    plt.title('F1-Score as function of NMF dimensions for different cases')
    plt.ylabel('f1 score')
    plt.xlabel('NMF embedding size')
    plt.errorbar(nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].n, nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].f1,
                 nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].f1_std, label='Low-Sparsity | Weighted NMF', color='r')
    plt.errorbar(nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].n, nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].f1,
                 nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].f1_std, label='Low-Sparsity | NMF', color='b')
    plt.errorbar(nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].n, nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].f1,
                 nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].f1_std, label='High-Sparsity | Weighted NMFt',
                 color='r',
                 linestyle='--')
    plt.errorbar(nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].n, nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].f1,
                 nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].f1_std, label='High-Sparsity| NMF', color='b',
                 linestyle='--')
    plt.legend(loc='upper left', title='case', prop={'size': 9})
    plt.savefig('./figures/nmf_f1_results.jpeg', bbox_inches="tight")
    print('(saved figure: nmf_f1_results.jpeg)')

    # plot all nmf acc results
    plt.figure()
    plt.title('Accuracy as function of NMF dimensions for different cases')
    plt.ylabel('accuracy score')
    plt.xlabel('NMF embedding size')
    plt.errorbar(nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].n, nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].acc,
                 nmf_res[(nmf_res.set == 'LS') & nmf_res.nan].acc_std, label='Low-Sparsity | Weighted NMF', color='r')
    plt.errorbar(nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].n, nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].acc,
                 nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].acc_std, label='Low-Sparsity | NMF', color='b')
    plt.errorbar(nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].n, nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].acc,
                 nmf_res[(nmf_res.set == 'HS') & nmf_res.nan].acc_std, label='High-Sparsity | Weighted NMF',
                 color='r',
                 linestyle='--')
    plt.errorbar(nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].n, nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].acc,
                 nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].acc_std, label='High-Sparsity | NMF', color='b',
                 linestyle='--')
    plt.legend(loc='upper left', title='case', prop={'size': 9})
    plt.savefig('./figures/nmf_acc_results.jpeg', bbox_inches="tight")
    print('(saved figure: nmf_acc_results.jpeg)')
    print('')

    print('--- saving plots for all MDS results ---')
    # plot all mds nmi results
    plt.figure()
    plt.title('Normalized mutual information as function of K-Means n_clusters for different cases')
    plt.ylabel('nmi score')
    plt.xlabel('K-Means n_clusters')
    colors = ['r', 'g', 'b', 'c']
    i = 0
    for n_embed in range(5, 25, 5):
        t_df = mds_res[mds_res.n_embed == n_embed]
        plt.plot(t_df[t_df.set == 'LS'].n_cluster, t_df[t_df.set == 'LS'].nmi,
                 label='Low_Spars | dim={}'.format(n_embed), color=colors[i])
        plt.plot(t_df[t_df.set == 'HS'].n_cluster, t_df[t_df.set == 'HS'].nmi,
                 label='High_Spars | dim={}'.format(n_embed), color=colors[i], linestyle='--')
        i += 1
    plt.legend(loc='best', title='case', prop={'size': 9})
    plt.savefig('./figures/mds_nmi_results.jpeg', bbox_inches="tight")
    print('(saved figure: mds_nmi_results.jpeg)')

    # plot all mds f1 results
    plt.figure()
    plt.title('F1 score as function of K-Means n_clusters for different cases')
    plt.ylabel('f1 score')
    plt.xlabel('K-Means n_clusters')
    colors = ['r', 'g', 'b', 'c']
    i = 0
    for n_embed in range(5, 25, 5):
        t_df = mds_res[mds_res.n_embed == n_embed]
        plt.errorbar(t_df[t_df.set == 'LS'].n_cluster, t_df[t_df.set == 'LS'].f1, t_df[t_df.set == 'LS'].f1_std,
                     label='Low_Spars | dim={}'.format(n_embed), color=colors[i])
        plt.errorbar(t_df[t_df.set == 'HS'].n_cluster, t_df[t_df.set == 'HS'].f1, t_df[t_df.set == 'HS'].f1_std,
                     label='High_Spars | dim={}'.format(n_embed), color=colors[i], linestyle='--')
        i += 1
    plt.legend(loc='best', title='case', prop={'size': 9})
    plt.savefig('./figures/mds_f1_results.jpeg', bbox_inches="tight")
    print('(saved figure: mds_f1_results.jpeg)')

    # plot all mds acc results
    plt.figure()
    plt.title('Accuracy score as function of K-Means n_clusters for different cases')
    plt.ylabel('accuracy score')
    plt.xlabel('K-Means n_clusters')
    colors = ['r', 'g', 'b', 'c']
    i = 0
    for n_embed in range(5, 25, 5):
        t_df = mds_res[mds_res.n_embed == n_embed]
        plt.errorbar(t_df[t_df.set == 'LS'].n_cluster, t_df[t_df.set == 'LS'].acc, t_df[t_df.set == 'LS'].acc_std,
                     label='Low_Spars | dim={}'.format(n_embed), color=colors[i])
        plt.errorbar(t_df[t_df.set == 'HS'].n_cluster, t_df[t_df.set == 'HS'].acc, t_df[t_df.set == 'HS'].acc_std,
                     label='High_Spars | dim={}'.format(n_embed), color=colors[i], linestyle='--')
        i += 1
    plt.legend(loc='best', title='case', prop={'size': 9})
    plt.savefig('./figures/mds_acc_results.jpeg', bbox_inches="tight")
    print('(saved figure: mds_acc_results.jpeg)')
    print('')

    print('--- Evaluating and comparing all performances ---')
    # plot best nmi
    print('-- Best nmi results --')
    plt.figure()
    N = 2
    idx_max_ls = nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].nmi.idxmax()
    idx_max_hs = nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].nmi.idxmax()
    best_nmf = (nmf_res.loc[idx_max_ls].nmi, nmf_res.loc[idx_max_hs].nmi)
    print('Highest NMI for nmf: Low-Sparsity {:.3f}, HighSparsity {:.3f}'.format(*best_nmf))
    print('With parameters: Low-Sparsity n={}, High-Sparsity n={}'.format(nmf_res.loc[idx_max_ls].n,
                                                                          nmf_res.loc[idx_max_hs].n, ))

    idx_max_ls = nmf_res[(nmf_res.set == 'LS') & (nmf_res.nan)].nmi.idxmax()
    idx_max_hs = nmf_res[(nmf_res.set == 'HS') & (nmf_res.nan)].nmi.idxmax()
    best_wnmf = (nmf_res.loc[idx_max_ls].nmi, nmf_res.loc[idx_max_hs].nmi)
    print('Highest NMI for wnmf: Low-Sparsity {:.3f}, High-Sparsity {:.3f}'.format(*best_wnmf))
    print('With parameters: Low-Sparsity n={}, High-Sparsity n={}'.format(nmf_res.loc[idx_max_ls].n,
                                                                          nmf_res.loc[idx_max_hs].n, ))

    idx_max_ls = mds_res[(mds_res.set == 'LS')].nmi.idxmax()
    idx_max_hs = mds_res[(mds_res.set == 'HS')].nmi.idxmax()
    best_mds = (mds_res.loc[idx_max_ls].nmi, mds_res.loc[idx_max_hs].nmi)
    print('Highest NMI for mds: Low-Sparsity {:.3f}, High-Sparaity {:.3f}'.format(*best_mds))
    print('With parameters: Low-Sparsity embed={} cluster={}, High-Sparsity  embed={} cluster={}'.format(
        mds_res.loc[idx_max_ls].n_embed, mds_res.loc[idx_max_ls].n_cluster, mds_res.loc[idx_max_hs].n_embed,
        mds_res.loc[idx_max_hs].n_cluster, ))

    ind = np.arange(N)
    plt.figure(figsize=(5, 4))
    width = 0.15
    plt.bar(ind, best_nmf, width, label='Traditional NMF')
    plt.bar(ind + width, best_wnmf, width, label='Weighted NMF')
    plt.bar(ind + 2 * width, best_mds, width, label='MDS + K-Means')
    plt.ylabel('best nmi score')
    plt.title('Best achieved NMI score for each set and algorithm')
    plt.xticks(ind + width, ('Low-Sparsity', 'High-Sparsity'))
    plt.legend(loc='best')
    plt.savefig('./figures/all_best_nmi_results.jpeg', bbox_inches="tight")
    print('(saved figure: all_best_nmi_results.jpeg)')

    # plot best f1
    print('-- Best f1 results --')
    plt.figure()
    N = 2
    idx_max_ls = nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].f1.idxmax()
    idx_max_hs = nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].f1.idxmax()
    best_nmf = (nmf_res.loc[idx_max_ls].f1, nmf_res.loc[idx_max_hs].f1)
    nmf_err = (nmf_res.loc[idx_max_ls].f1_std, nmf_res.loc[idx_max_hs].f1_std)
    print('Highest f1 for nmf: Low-Sparsity {:.3f} \u00B1 {:.3f}, HighSparsity {:.3f} \u00B1 {:.3f}'.format(best_nmf[0],
                                                                                                            nmf_err[0],
                                                                                                            best_nmf[1],
                                                                                                            nmf_err[1]))
    print('With parameters: Low-Sparsity n={}, High-Sparsity n={}'.format(nmf_res.loc[idx_max_ls].n,
                                                                          nmf_res.loc[idx_max_hs].n, ))

    idx_max_ls = nmf_res[(nmf_res.set == 'LS') & (nmf_res.nan)].f1.idxmax()
    idx_max_hs = nmf_res[(nmf_res.set == 'HS') & (nmf_res.nan)].f1.idxmax()
    best_wnmf = (nmf_res.loc[idx_max_ls].f1, nmf_res.loc[idx_max_hs].f1)
    wnmf_err = (nmf_res.loc[idx_max_ls].f1_std, nmf_res.loc[idx_max_hs].f1_std)
    print(
        'Highest f1 for wnmf: Low-Sparsity {:.3f} \u00B1 {:.3f}, HighSparsity {:.3f} \u00B1 {:.3f}'.format(best_wnmf[0],
                                                                                                           wnmf_err[0],
                                                                                                           best_wnmf[1],
                                                                                                           wnmf_err[1]))
    print('With parameters: Low-Sparsity n={}, High-Sparsaity n={}'.format(nmf_res.loc[idx_max_ls].n,
                                                                           nmf_res.loc[idx_max_hs].n, ))

    idx_max_ls = mds_res[mds_res.set == 'LS'].f1.idxmax()
    idx_max_hs = mds_res[mds_res.set == 'HS'].f1.idxmax()
    best_mds = (mds_res.loc[idx_max_ls].f1, mds_res.loc[idx_max_hs].f1)
    mds_err = (mds_res.loc[idx_max_ls].f1_std, mds_res.loc[idx_max_hs].f1_std)
    print('Highest f1 for mds: Low-Sparsity {:.3f} \u00B1 {:.3f}, HighSparsity {:.3f} \u00B1 {:.3f}'.format(best_mds[0],
                                                                                                            mds_err[0],
                                                                                                            best_mds[1],
                                                                                                            mds_err[1]))
    print('With parameters: Low-Sparsity embed={} cluster={}, High-Sparsity  embed={} cluster={}'.format(
        mds_res.loc[idx_max_ls].n_embed, mds_res.loc[idx_max_ls].n_cluster, mds_res.loc[idx_max_hs].n_embed,
        mds_res.loc[idx_max_hs].n_cluster, ))

    best_rand = (rand['rand_f1'], rand['rand_f1'])
    rand_err = (rand['rand_f1_err'], rand['rand_f1_err'])
    print('Highest f1 for nmf:{:.3f} \u00B1 {:.3f}'.format(best_rand[0], rand_err[0]))
    ind = np.arange(N)
    plt.figure(figsize=(5, 4))
    width = 0.15
    plt.bar(ind, best_nmf, width, yerr=nmf_err, label='Traditional NMF')
    plt.bar(ind + width, best_wnmf, width, yerr=wnmf_err, label='Weighted NMF')
    plt.bar(ind + 2 * width, best_mds, width, yerr=mds_err, label='MDS + K-Means')
    plt.bar(ind + 3 * width, best_rand, width, yerr=rand_err, label='Random')
    plt.ylabel('best F1 score')
    plt.title('Best achieved F1 score for each set and algorithm')
    plt.xticks(ind + 1.5 * width, ('Low-Sparsity', 'High-Sparsity'))
    plt.legend(loc='best')
    plt.savefig('./figures/all_best_f1_results.jpeg', bbox_inches="tight")
    print('(saved figure: all_best_f1_results.jpeg)')

    # plot best acc
    print('-- Best acc results --')
    plt.figure()
    N = 2
    idx_max_ls = nmf_res[(nmf_res.set == 'LS') & (~nmf_res.nan)].acc.idxmax()
    idx_max_hs = nmf_res[(nmf_res.set == 'HS') & (~nmf_res.nan)].acc.idxmax()
    best_nmf = (nmf_res.loc[idx_max_ls].acc, nmf_res.loc[idx_max_hs].acc)
    nmf_err = (nmf_res.loc[idx_max_ls].acc_std, nmf_res.loc[idx_max_hs].acc_std)
    print('Highest accuracy for nmf: Low-Sparsity {:.3f} \u00B1 {:.3f}, HighSparsity {:.3f} \u00B1 {:.3f}'.format(
        best_nmf[0], nmf_err[0], best_nmf[1], nmf_err[1]))
    print('With parameters: Low-Sparsity n={}, High-Sparsity n={}'.format(nmf_res.loc[idx_max_ls].n,
                                                                          nmf_res.loc[idx_max_hs].n, ))

    idx_max_ls = nmf_res[(nmf_res.set == 'LS') & (nmf_res.nan)].acc.idxmax()
    idx_max_hs = nmf_res[(nmf_res.set == 'HS') & (nmf_res.nan)].acc.idxmax()
    best_wnmf = (nmf_res.loc[idx_max_ls].acc, nmf_res.loc[idx_max_hs].acc)
    wnmf_err = (nmf_res.loc[idx_max_ls].acc_std, nmf_res.loc[idx_max_hs].acc_std)
    print('Highest accuracy for wnmf: Low-Sparsity {:.3f} \u00B1 {:.3f}, HighSparsity {:.3f} \u00B1 {:.3f}'.format(
        best_wnmf[0], wnmf_err[0], best_wnmf[1], wnmf_err[1]))
    print('With parameters: Low-Sparsity n={}, High-Sparsity n={}'.format(nmf_res.loc[idx_max_ls].n,
                                                                          nmf_res.loc[idx_max_hs].n, ))

    idx_max_ls = mds_res[mds_res.set == 'LS'].acc.idxmax()
    idx_max_hs = mds_res[mds_res.set == 'HS'].acc.idxmax()
    best_mds = (mds_res.loc[idx_max_ls].acc, mds_res.loc[idx_max_hs].acc)
    mds_err = (mds_res.loc[idx_max_ls].acc_std, mds_res.loc[idx_max_hs].acc_std)
    print('Highest accuracy for mds: Low-Sparsity {:.3f} \u00B1 {:.3f}, HighSparsity {:.3f} \u00B1 {:.3f}'.format(
        best_mds[0], mds_err[0], best_mds[1], mds_err[1]))
    print('With parameters: Low-Sparsity embed={} cluster={}, High-Sparsity  embed={} cluster={}'.format(
        mds_res.loc[idx_max_ls].n_embed, mds_res.loc[idx_max_ls].n_cluster, mds_res.loc[idx_max_hs].n_embed,
        mds_res.loc[idx_max_hs].n_cluster, ))

    best_rand = (rand['rand_acc'], rand['rand_acc'])
    rand_err = (rand['rand_acc_err'], rand['rand_acc_err'])

    ind = np.arange(N)
    plt.figure(figsize=(5, 4))
    width = 0.15
    plt.bar(ind, best_nmf, width, yerr=nmf_err, label='Traditional NMF')
    plt.bar(ind + width, best_wnmf, width, yerr=wnmf_err, label='Weighted NMF')
    plt.bar(ind + 2 * width, best_mds, width, yerr=mds_err, label='MDS + K-Means')
    plt.bar(ind + 3 * width, best_rand, width, yerr=rand_err, label='Random')
    plt.ylabel('best accuracy score')
    plt.title('Best achieved accuracy score for each set and algorithm')
    plt.xticks(ind + 1.5 * width, ('Low-Sparsity', 'High-Sparsity'))
    plt.legend(loc='best')
    plt.savefig('./figures/all_best_accuracy_results.jpeg', bbox_inches="tight")
    print('(saved figure: all_best_acc_results.jpeg)')


if __name__ == "__main__":
    main()
