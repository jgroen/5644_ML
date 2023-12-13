from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.cluster import HDBSCAN
from sklearn.cluster import OPTICS
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_cluster(kmeans, name, data, labels):
    """Benchmark to evaluate the clustering methods.

    Parameters
    ----------
    kmeans : model instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


path = './project_data/'

all_files = glob.glob(os.path.join(path, "*.csv"))

dfs = []
for f in all_files:
    #print(f)
    f_name = os.path.splitext(os.path.basename(f))[0]
    #print(f_name)
    data = pd.read_csv(f, header=0)
    data['Label'] = f_name
    dfs.append(data)

frame = pd.concat(dfs, ignore_index=True)
frame[frame.select_dtypes(['object']).columns] = frame.select_dtypes(['object']).apply(lambda x: x.astype('category').cat.codes)
frame = frame.fillna(0)

# print(frame)
# print(frame.dtypes)

in_data = frame[["Time", "Direction", "Protocol", "Length"]].to_numpy()
# print(in_data.shape[0])

# kmeans = KMeans(n_clusters=3, n_init='auto').fit(in_data)
# print(kmeans.labels_)

# ******** initial testing follows ******** #
#
# print('Testing with window size 1 \n')
# print(82 * "_")
# print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette")
#
# kmeans = KMeans(init='k-means++', n_clusters=3, n_init='auto')
# bench_cluster(kmeans=kmeans, name='K-Means++', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# kmeans = KMeans(init='random', n_clusters=3, n_init='auto')
# bench_cluster(kmeans=kmeans, name='random', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# pca = PCA(n_components=3).fit(in_data)
# kmeans = KMeans(init=pca.components_, n_clusters=3, n_init=1)
# bench_cluster(kmeans=kmeans, name='PCA', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# kmeans = BisectingKMeans(n_clusters=3)
# bench_cluster(kmeans=kmeans, name='Bisecting', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
# print(82 * "_")
#
# print(82 * "_")
# print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette")
#
# kmeans = KMeans(init='k-means++', n_clusters=2, n_init='auto')
# bench_cluster(kmeans=kmeans, name='K-Means++', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# kmeans = KMeans(init='random', n_clusters=2, n_init='auto')
# bench_cluster(kmeans=kmeans, name='random', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# pca = PCA(n_components=2).fit(in_data)
# kmeans = KMeans(init=pca.components_, n_clusters=2, n_init=1)
# bench_cluster(kmeans=kmeans, name='PCA', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# kmeans = BisectingKMeans(n_clusters=2)
# bench_cluster(kmeans=kmeans, name='Bisecting', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
# print(82 * "_")
#
#
# print(82 * "_")
# t0 = time()
# hdbscan = HDBSCAN(min_cluster_size=5, n_jobs=-1).fit(in_data)
# fit_time = time()-t0
# print('HDBSCAN creates {} total clusters in {}s'.format(max(hdbscan.labels_), fit_time))
# print(82 * "_")
#
# # spectral = SpectralClustering(n_clusters=3).fit(in_data)
# # bench_cluster(kmeans=spectral, name='Spectral', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# # ward = AgglomerativeClustering(n_clusters=3)
# # bench_cluster(kmeans=ward, name='Ward', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# # kmeans = MeanShift(bandwidth=2)
# # bench_cluster(kmeans=kmeans, name='B=2', data=in_data, labels=frame[["Label"]].to_numpy().flatten())
#
# print(82 * "_")
# x = 10
# while x <= 10**3:
#     t0 = time()
#     optics = OPTICS(max_eps=x).fit(in_data)
#     fit_time = time()-t0
#     print('OPTICS with max_eps of {} creates {} total clusters in {}s'.format(x, max(optics.labels_), fit_time))
#     x = x * 10
#
# print(82 * "_")
# #

# ******** advanced testing follows ******** #

def cluster_test(in_data):
    print(82 * "_")

    x = 10
    while x <= 10 ** 1:
        metrics = ['minkowski']
        min_samples = [25]
        xi_list = [0.05]
        for m in metrics:
            for ms in min_samples:
                for xi in xi_list:
                    t0 = time()
                    optics = OPTICS(max_eps=x, n_jobs=-1, metric=m, min_samples=ms, xi=xi).fit(in_data)
                    # metric [‘cosine’, ‘euclidean’, ‘l1’, ‘l2’]
                    # min_samples [10, 15, 20, 25]
                    # xi [0.1, 0.15, 0.2, 0.25]
                    fit_time = time() - t0
                    print('OPTICS eps={} m={} ms={} xi={} creates {} total clusters in {}s'.format(x, m, ms, xi,
                                                                                    max(optics.labels_), fit_time))
        x = x * 10
    print(82 * "_")

    alpha = [1.2]
    cluster = [16]
    mcs = [25]

    for a in alpha:
        for c in cluster:
            for m in mcs:
                t0 = time()
                hdbscan = HDBSCAN(min_cluster_size=m, n_jobs=-1, allow_single_cluster=True, alpha=a,
                                  cluster_selection_epsilon=c).fit(in_data)
                # alpha [.8 .9 1 1.1 1.2]
                # cluster_selection_epsilon [1 2 4 8 16]
                # min_cluster_size [5 10 15 20 25]
                fit_time = time() - t0
                print('HDBSCAN a={} c={} m={} creates {} total clusters in {}s'.format(a, c, m, max(hdbscan.labels_),
                                                                                       fit_time))

    print(82 * "_")


# repeat with sliding window sizes
old_data = in_data
# print(old_data.shape)
# window_size = [128, 136, 184, 192, 198, 224, 230, 256, 264, 272]
window_size = [120, 128, 136, 160]
for w in window_size:
    in_data = np.zeros([old_data.shape[0]+1-w, 4*w])
    l = 1
    while l <= in_data.shape[0]+1-w:
        # print(old_data[l, :])
        # print(old_data[l+1, :])
        in_data[l, :] = old_data[l:l+w, :].flatten()
        l = l + 1


    print('\nTesting with window size {}'.format(w))
    print('input data size {}'.format(in_data.shape))
    cluster_test(in_data)


