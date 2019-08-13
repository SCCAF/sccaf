#  __init__.py
#
#  Copyright 2018 Chichau Miau <zmiao@ebi.ac.uk>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import pandas as pd
import numpy as np
from scipy.sparse import issparse
from pandas.api.types import is_categorical_dtype
from collections import defaultdict
import louvain
import scipy
import os
import seaborn as sns
import patsy

#'from anndata import AnnData
from scanpy import settings as sett
from scanpy import logging as logg

import scanpy as sc
# for color
from scanpy.plotting.palettes import *

import matplotlib.pyplot as plt
from matplotlib import rcParams

# for reading/saving clf model
from sklearn.mixture import BayesianGaussianMixture
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

color_long = ['#e6194b',
              '#3cb44b',
              '#ffe119',
              '#0082c8',
              '#f58231',
              '#911eb4',
              '#46f0f0',
              '#f032e6',
              '#d2f53c',
              '#fabebe',
              '#008080',
              '#e6beff',
              '#aa6e28',
              '#800000',
              '#aaffc3',
              '#808000',
              '#ffd8b1',
              '#000080',
              '#808080',
              '#000000', ] + default_26


def run_BayesianGaussianMixture(Y, K):
    """
    For K-means clustering

    Input
    -----
    Y: the expression matrix
    K: number of clusters

    return
    -----
    clusters assigned to each cell.
    """
    gmm = BayesianGaussianMixture(K, max_iter=1000)
    gmm.fit(Y)
    return gmm.predict(Y)


def bhattacharyya_distance(repr1, repr2):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = - np.log(np.sum([np.sqrt(p * q) for (p, q) in zip(repr1, repr2)]))
    assert not np.isnan(sim), 'Error: Similarity is nan.'
    if np.isinf(sim):
        # the similarity is -inf if no term in the review is in the vocabulary
        return 0
    return sim


def bhattacharyya_matrix(prob, flags=None):
    ndim = prob.shape[1]
    m = np.zeros((ndim, ndim))
    if flags is None:
        for i in np.arange(ndim):
            for j in np.arange(i + 1, ndim):
                val = -bhattacharyya_distance(prob[:, i], prob[:, j])
                m[i, j] = val
                m[j, i] = val
    else:
        for i in np.arange(ndim):
            for j in np.arange(i + 1, ndim):
                df = pd.DataFrame({'i': prob[:, i], 'j': prob[:, j], 'f': flags})
                df = df[df['f']]
                val = -bhattacharyya_distance(df['i'], df['j'])
                m[i, j] = val
                m[j, i] = val
    return m


def binary_accuracy(X, y, clf):
    y_pred = clf.predict(X)
    return y == y_pred


def normalize_confmat1(cmat, mod='1'):
    """
    Normalize the confusion matrix based on the total number of cells in each class
    x(i,j) = max(cmat(i,j)/diagnol(i),cmat(j,i)/diagnol(j))
    confusion rate between i and j is defined by the maximum ratio i is confused as j or j is confused as i.

    Input
    cmat: the confusion matrix

    return
    -----
    the normalized confusion matrix
    """
    dmat = cmat.values
    smat = np.diag(dmat)
    dim = cmat.shape[0]
    xmat = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(i + 1, dim):
            if mod is '1':
                xmat[i, j] = xmat[j, i] = max(dmat[i, j] / smat[j], dmat[j, i] / smat[i])
            else:
                xmat[i, j] = xmat[j, i] = max(dmat[i, j] / smat[i], dmat[j, i] / smat[j])
    return xmat


def normalize_confmat2(cmat):
    """
    Normalize the confusion matrix based on the total number of cells.
    x(i,j) = max(cmat(i,j)+cmat(j,i)/N)
    N is total number of cells analyzed.
    Confusion rate between i and j is defined by the sum of i confused as j or j confused as i.
    Then divide by total number of cells.

    Input
    cmat: the confusion matrix

    return
    -----
    the normalized confusion matrix
    """
    emat = np.copy(cmat)
    s = np.sum(cmat)
    emat = emat + emat.T
    np.fill_diagonal(emat, 0)
    return emat * 1.0 / s


def cluster_adjmat(xmat,
                   resolution=1,
                   cutoff=0.1):
    """
    Cluster the groups based on the adjacent matrix.
    Use the cutoff to discretize the matrix used to construct the adjacent graph.
    Then cluster the graph using the louvain clustering with a resolution value.
    As the adjacent matrix is binary, the default resolution value is set to 1.

    Input
    -----
    xmat: `numpy.array` or sparse matrix
        the reference matrix/normalized confusion matrix
    cutoff: `float` optional (default: 0.1)
        threshold used to binarize the reference matrix
    resolution: `float` optional (default: 1.0)
        resolution parameter for louvain clustering

    return
    -----
    new group names.
    """
    g = sc.utils.get_igraph_from_adjacency((xmat > cutoff).astype(int), directed=False)
    print(g)
    part = louvain.find_partition(g, louvain.RBConfigurationVertexPartition,
                                  resolution_parameter=resolution)
    groups = np.array(part.membership)
    return groups


def msample(x, n, frac):
    """
    sample the matrix by number or by fraction.
    if the fraction is larger than the sample number, use number for sampling. Otherwise, use fraction.

    Input
    -----
    x: the matrix to be split
    n: number of vectors to be sampled
    frac: fraction of the total matrix to be sampled

    return
    -----
    sampled selection.
    """
    if len(x) <= np.floor(n / frac):
        if len(x) < 10: frac = 0.9
        return x.sample(frac=frac)
    else:
        return x.sample(n=n)


def train_test_split_per_type(X, y, n=100, frac=0.8):
    """
    This function is identical to train_test_split, but can split the data either based on number of cells or by fraction.

    Input
    -----
    X: `numpy.array` or sparse matrix
        the feature matrix
    y: `list of string/int`
        the class assignments
    n: `int` optional (default: 100)
        maximum number sampled in each label
    fraction: `float` optional (default: 0.8)
        Fraction of data included in the training set. 0.5 means use half of the data for training,
        if half of the data is fewer than maximum number of cells (n).

    return
    -----
    X_train, X_test, y_train, y_test
    """
    df = pd.DataFrame(y)
    df.index = np.arange(len(y))
    df.columns = ['class']
    c_idx = df.groupby('class').apply(lambda x: msample(x, n=n, frac=frac)).index.get_level_values(None)
    d_idx = ~np.isin(np.arange(len(y)), c_idx)
    return X[c_idx, :], X[d_idx, :], y[c_idx], y[d_idx]


# functions for SCCAF
def SCCAF_assessment(*args, **kwargs):
    """
    Assessment of clustering reliability using self-projection.
    It is the same as the self_projection function.
    """
    return self_projection(*args, **kwargs)


# need to check number of cells in each cluster of the training set.
def self_projection(X, 
                    cell_types,
                    classifier="LR",
                    penalty='l1',
                    sparsity=0.5,
                    fraction=0.5,
                    random_state=1,
                    n=0,
                    cv=5,
                    whole=False, 
                    n_jobs=None):
    # n = 100 should be good.
    """
    This is the core function for running self-projection.

    Input
    -----
    X: `numpy.array` or sparse matrix
        the expression matrix, e.g. ad.raw.X.
    cell_types: `list of String/int`
        the cell clustering assignment
    classifier: `String` optional (defatul: 'LR')
        a machine learning model in "LR" (logistic regression), \
        "RF" (Random Forest), "GNB"(Gaussion Naive Bayes), "SVM" (Support Vector Machine) and "DT"(Decision Tree).
    penalty: `String` optional (default: 'l2')
        the standardization mode of logistic regression. Use 'l1' or 'l2'.
    sparsity: `fload` optional (default: 0.5)
        The sparsity parameter (C in sklearn.linear_model.LogisticRegression) for the logistic regression model.
    fraction: `float` optional (default: 0.5)
        Fraction of data included in the training set. 0.5 means use half of the data for training,
        if half of the data is fewer than maximum number of cells (n).
    random_state: `int` optional (default: 1)
        random_state parameter for logistic regression.
    n: `int` optional (default: 100)
        Maximum number of cell included in the training set for each cluster of cells.
        only fraction is used to split the dataset if n is 0.
    cv: `int` optional (default: 5)
        fold for cross-validation on the training set.
        0 means no cross-validation.
    whole: `bool` optional (default: False)
        if measure the performance on the whole dataset (include training and test).
    n_jobs: `int` optional, number of threads to use with the different classifiers (default: None - unlimited).

    return
    -----
    y_prob, y_pred, y_test, clf
    y_prob: `matrix of float`
        prediction probability
    y_pred: `list of string/int`
        predicted clustering of the test set
    y_test: `list of string/int`
        real clustering of the test set
    clf: the classifier model.
    """
    # split the data into training and testing
    if n > 0:
        X_train, X_test, y_train, y_test = \
            train_test_split_per_type(X, cell_types, n=n, frac=(1 - fraction))
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, cell_types,
                             stratify=cell_types, test_size=fraction)  # fraction means test size
    # set the classifier
    if classifier == 'LR':
        clf = LogisticRegression(random_state=1, penalty=penalty, C=sparsity, n_jobs=n_jobs)
    elif classifier == 'RF':
        clf = RandomForestClassifier(random_state=1, n_jobs=n_jobs)
    elif classifier == 'GNB':
        clf = GaussianNB()
    elif classifier == 'GPC':
        clf = GaussianProcessClassifier(n_jobs=n_jobs)
    elif classifier == 'SVM':
        clf = SVC(probability=True)
    elif classifier == 'SH':
        clf = SGDClassifier(loss='squared_hinge', n_jobs=n_jobs)
    elif classifier == 'PCP':
        clf = SGDClassifier(loss='perceptron', n_jobs=n_jobs)
    elif classifier == 'DT':
        clf = DecisionTreeClassifier()
    
    # mean cross validation score
    cvsm = 0
    if cv > 0:
        cvs = cross_val_score(clf, X_train, np.array(y_train), cv=cv, scoring='accuracy', n_jobs=n_jobs)
        cvsm = cvs.mean()
        print("Mean CV accuracy: %.4f" % cvsm)
    # accuracy on cross validation and on test set
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    print("Accuracy on the training set: %.4f" % accuracy)
    accuracy_test = clf.score(X_test, y_test)
    print("Accuracy on the hold-out set: %.4f" % accuracy_test)
    
    # accuracy of the whole dataset
    if whole:
        accuracy = clf.score(X, cell_types)
        print("Accuracy on the whole set: %.4f" % accuracy)
    
    # get predicted probability on the test set
    y_prob = None
    if not classifier in ['SH', 'PCP']:
        y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    
    return y_prob, y_pred, y_test, clf, cvsm, accuracy_test


def make_unique(dup_list):
    """
    Make a name list unique by adding suffix "_%d". This function is identical to the make.unique function in R.

    Input
    -----
    dup_list: a list

    return
    -----
    a unique list with the same length as the input.
    """
    from collections import Counter

    counter = Counter()
    deduped = []
    for name in dup_list:
        new = str(name) + "_%s" % str(counter[name]) if counter[name] else name
        counter.update({name: 1})
        deduped.append(new)
    return deduped


def confusion_matrix(y_test, y_pred, clf, labels=None):
    """
    Get confusion matrix based on the test set.

    Input
    -----
    y_test, y_pred, clf: same as in self_projection

    return
    -----
    the confusion matrix
    """
    if labels is None: labels = clf.classes_
    df = pd.DataFrame.from_records(metrics.confusion_matrix(y_test, y_pred, labels=labels))
    df.index = labels
    df.columns = labels
    df.index.name = 'Real'
    df.columns.name = 'Predicted'
    return df


def per_cluster_accuracy(mtx, ad=None, clstr_name='louvain'):
    """
    Measure the accuracy of each cluster and put into a metadata slot.
    So the reliability of each cluster can be visualized.

    Input
    -----
    mtx: `pandas.dataframe`
        the confusion matrix
    ad: `AnnData`
        anndata object
    clstr_name: `String`
        the name of the clustering

    return
    -----
    """
    ds = None
    if not ad is None:
        ds = (np.diag(mtx.values) / mtx.sum(0)).fillna(0)
        rel_name = '%s_reliability' % clstr_name
        ad.obs[rel_name] = ad.obs[clstr_name].astype('category')
        ad.obs[rel_name].cat.categories = make_unique(ds)
        ad.obs[rel_name] = ad.obs[rel_name].astype(str).str.split("_").str[0]
        ad.obs[rel_name] = ad.obs[rel_name].astype(float)
    return ds


def per_cell_accuracy(X, cell_types, clf):
    y_prob = clf.predict_proba(X)
    df = pd.DataFrame(y_prob, index=cell_types, columns=clf.classes_).sort_index().T
    df.index.name = 'Predicted'
    dy = np.empty([0])
    for cell in df.index:
        x = np.array(df.loc[cell][cell].values)
        dy = np.concatenate((dy, x))
    return dy/np.max(df.values, 0)


def get_topmarkers(clf, names, topn=10):
    """
    Get the top weighted features from the logistic regressioin model.

    Input
    -----
    clf: the logistic regression classifier
    names: `list of Strings`
        the names of the features (the gene names).
    topn: `int`
        number of top weighted featured to be returned.

    return
    -----
    list of markers for each of the cluster.
    """
    marker_genes = pd.DataFrame({
        'cell_type': clf.classes_[clf.coef_.argmax(0)],
        'gene': names,
        'weight': clf.coef_.max(0)
    })

    top_markers = \
        marker_genes \
            .query('weight > 0.') \
            .sort_values('weight', ascending=False) \
            .groupby('cell_type') \
            .head(topn) \
            .sort_values(['cell_type', 'weight'], ascending=[True, False])
    return top_markers

def eu_distance(X, gp1, gp2, cell):
    """
    Measure the euclidean distance between two groups of cells and the third group.

    Input
    -----
    X: `np.array` or `sparse matrix`
        the total expression matrix
    gp1: `bool list`
        group1 of cells
    gp2: `bool list`
        group2 of cells
    cell: `bool list`
        group3 of cells, the group to be compared with gp1 and gp2.

    return
    -----
    `float value`
    the average distance difference.
    """
    d1 = euclidean_distances(X[gp1 & (~cell), :], X[cell, :])
    d2 = euclidean_distances(X[gp2 & (~cell), :], X[cell, :])
    df1 = pd.DataFrame(d1[:, 0], columns=['distance'])
    df1['type'] = 'cell'
    df2 = pd.DataFrame(d2[:, 0], columns=['distance'])
    df2['type'] = 'cell_pred'
    df = pd.concat([df1, df2])
    m1 = d1.mean()
    m2 = d2.mean()
    print('%f - %f' % (m1, m2))
    return df


def get_distance_matrix(X, clusters, labels=None, metric='euclidean'):
    """
    Get the mean distance matrix between all clusters.

    Input
    -----
    X: `np.array` or `sparse matrix`
        the total expression matrix
    clusters: `string list`
        the assignment of the clusters
    labels: `string list`
        the unique labels of the clusters
    metric: `string` (optional, default: euclidean)
        distance metrics, see (http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)

    return
    -----
    `np.array`
        the all-cluster to all-cluster distance matrix.
    """
    if labels is None:
        labels = np.unique(clusters)
    centers = []
    if scipy.sparse.issparse(X):
        for cl in labels:
            centers.append(np.array(X[np.where(clusters == cl)[0], :].mean(0))[0, :])
    else:
        for cl in labels:
            centers.append(np.array(X[np.where(clusters == cl)[0], :].mean(0)))
    return pairwise_distances(np.array(centers), metric=metric)

def merge_cluster(ad, old_id, new_id, groups):
    ad.obs[new_id] = ad.obs[old_id]
    ad.obs[new_id] = ad.obs[new_id].astype('category')
    ad.obs[new_id].cat.categories = make_unique(groups.astype(str))
    ad.obs[new_id] = ad.obs[new_id].str.split('_').str[0]
    return ad
    
    
def find_high_resolution(ad, resolution=4, n=100):
    cut = resolution
    while cut > 0.5:
        print("clustering with resolution: %.1f" % cut)
        sc.tl.leiden(ad, resolution=cut)
        ad.obs['leiden_res%.1f' % cut] = ad.obs['leiden']
        if ad.obs['leiden'].value_counts().min() > n:
            break
        cut -= 0.5


def get_connection_matrix(ad_obs, key1, key2):
    df = ad_obs.groupby([key1,key2]).size().to_frame().reset_index()
    df.columns = [key1,key2,'counts']
    df2 = ad_obs[key2].value_counts()
    df['size'] = df2[df[key2].tolist()].tolist()
    df['percent'] = df['counts']/df['size']
    df = df[df['percent']>0.1]
    df2 = df.groupby(key1).size()
    df2 = df2[df2>1]
    df = df[df[key1].isin(df2.index)]

    dim = len(ad_obs[key2].unique())
    mat = pd.DataFrame(np.zeros([dim,dim])).astype(int)
    mat.columns = mat.columns.astype(str)
    mat.index = mat.index.astype(str)

    import itertools
    grouped = df.groupby(key1)
    for name, group in grouped:
        x = group[key2]
        if len(x)>0:
            for i,j in list(itertools.combinations(x.tolist(), 2)):
                mat.loc[i,j] = mat.loc[j,i] = 1
    return mat    


def SCCAF_optimize_all(ad,
                       min_acc=0.9,
                       R1norm_cutoff=0.5,
                       R2norm_cutoff=0.05,
                       R1norm_step=0.01,
                       R2norm_step=0.001, 
                       prefix='L1',
                       min_i = 3,
                       *args, **kwargs):
    """
    ad: `AnnData`
        The AnnData object of the expression profile.
    min_acc: `float` optional (default: 0.9)
        The minimum self-projection accuracy to be optimized for.
        e.g., 0.9 means the clustering optimization (merging process)
        will not stop until the self-projection accuracy is above 90%.
    R1norm_cutoff: `float` optional (default: 0.5)
        The start cutoff for R1norm of confusion matrix.
        e.g., 0.5 means use 0.5 as a cutoff to discretize the confusion matrix after R1norm.
        the discretized matrix is used to construct the connection graph for clustering optimization.
    R2norm_cutoff: `float` optional (default: 0.05)
        The start cutoff for R2norm of confusion matrix.
    R1norm_step: `float` optional (default: 0.01)
        The reduce step for minimum R1norm value.
        Each round of optimization calls the function `SCCAF_optimize`.
        The start of the next round of optimization is based on a new
        cutoff for the R1norm of the confusion matrix. This cutoff is
        determined by the minimum R1norm value in the previous round minus the R1norm_step value.
    R2norm_step: `float` optional (default: 0.001)
        The reduce step for minimum R2norm value.
    """
    acc = 0
    start_iter = 0
    
    clstr_old = len(ad.obs['%s_Round%d'%(prefix, start_iter)].unique())
    #'while acc < min_acc:
    for i in range(10):
        print("start_iter: %d" % start_iter)
        print("R1norm_cutoff: %f" % R1norm_cutoff)
        print("R2norm_cutoff: %f" % R2norm_cutoff)
        print("Accuracy: %f" % acc)
        print("======================")
        ad, m1, m2, acc, start_iter = SCCAF_optimize(ad=ad,
                                                     R1norm_cutoff=R1norm_cutoff,
                                                     R2norm_cutoff=R2norm_cutoff,
                                                     start_iter=start_iter,
                                                     min_acc=min_acc, 
                                                     prefix=prefix,
                                                     *args, **kwargs)
        print("m1: %f" % m1)
        print("m2: %f" % m2)
        print("Accuracy: %f" % acc)
        R1norm_cutoff = m1 - R1norm_step
        R2norm_cutoff = m2 - R2norm_step
        
        clstr_new = len(ad.obs['%s_result'%prefix].unique())
        
        if clstr_new >= clstr_old and i >= min_i:
            print("converged SCCAF_all!")
            break
        
        if acc >=min_acc:
            break
            
            
def SCCAF_optimize(ad,
                   prefix='L1',
                   use='raw',
                   use_projection=False,
                   R1norm_only=False,
                   R2norm_only=False,
                   dist_only=False,
                   dist_not=True,
                   plot=True,
                   basis='umap',
                   plot_dist=False,
                   plot_cmat=False,
                   mod='1',
                   low_res = None,
                   c_iter=3,
                   n_iter=10,
                   n_jobs=None,
                   start_iter=0,
                   sparsity=0.5,
                   n=100,
                   fraction=0.5,
                   R1norm_cutoff=0.1,
                   R2norm_cutoff=1,
                   dist_cutoff=8,
                   classifier="LR", 
                   min_acc=1):
    """
    This is a self-projection confusion matrix directed cluster optimization function.

    Input
    -----
    ad: `AnnData`
        The AnnData object of the expression profile.
    prefix: `String`, optional (default: 'L1')
        The name of the optimization, which set as a prefix.
        e.g., the prefix = 'L1', the start round of optimization clustering is based on
        'L1_Round0'. So we need to assign an over-clustering state as a start point.
        e.g., ad.obs['L1_Round0'] = ad.obs['louvain']
    use: `String`, optional (default: 'raw')
        Use what features to train the classifier. Three choices:
        'raw' uses all the features;
        'hvg' uses the highly variable genes in the anndata object ad.var_names slot;
        'pca' uses the PCA data in the anndata object ad.obsm['X_pca'] slot.
    R1norm_only: `bool` optional (default: False)
        If only use the confusion matrix(R1norm) for clustering optimization.
    R2norm_only: `bool` optional (default: False)
        If only use the confusion matrix(R2norm) for clustering optimization.
    dist_only: `bool` optional (default: False)
        If only use the distance matrix for clustering optimization.
    dist_not: `bool` optional (default: True)
        If not use the distance matrix for clustering optimization.
    plot: `bool` optional (default: True)
        If plot the self-projectioin results, ROC curves and confusion matrices,
        during the optimization.
    plot_tsne: `bool` optional (default: False)
        If plot the self-projectioin results as tSNE. If False, the results are plotted as UMAP.
    plot_dist: `bool` optional (default: False)
        If make a scatter plot of the distance compared with the confusion rate for each of the cluster.
    plot_cmat: `bool` optional (default: False)
        plot the confusion matrix or not.
    mod: `string` optional (default: '1')
        two directions of normalization of confusion matrix for R1norm.
    c_iter: `int` optional (default: 3)
        Number of iterations of sampling for the confusion matrix.
        The minimum value of confusion rate in all the iterations is used as the confusion rate between two clusters.
    n_iter： `int` optional (default: 10)
        Maximum number of iterations(Rounds) for the clustering optimization.
    start_iter： `int` optional (default: 0)
        The start round of the optimization. e.g., start_iter = 3,
        the optimization will start from ad.obs['%s_3'%prefix].
    sparsity: `fload` optional (default: 0.5)
        The sparsity parameter (C in sklearn.linear_model.LogisticRegression) for the logistic regression model.
    n: `int` optional (default: 100)
        Maximum number of cell included in the training set for each cluster of cells.
    n_jobs: `int` number of jobs/threads to use (default: None - unlimited).
    fraction: `float` optional (default: 0.5)
        Fraction of data included in the training set. 0.5 means use half of the data for training,
        if half of the data is fewer than maximum number of cells (n).
    R1norm_cutoff: `float` optional (default: 0.1)
        The cutoff for the confusion rate (R1norm) between two clusters.
        0.1 means we allow maximum 10% of the one cluster confused as another cluster.
    R2norm_cutoff: `float` optional (default: 1.0)
        The cutoff for the confusion rate (R2norm) between two clusters.
        1.0 means the confusion between any two cluster should not exceed 1% of the total number of cells.
    dist_cutoff: `float` optional (default: 8.0)
        The cutoff for the euclidean distance between two clusters of cells.
        8.0 means the euclidean distance between two cell types should be greater than 8.0.
    low_res: `str` optional
		the clustering boundary for under-clustering. Set a low resolution in louvain/leiden clustering and give
		the key as the underclustering boundary.
    classifier: `String` optional (default: 'LR')
        a machine learning model in "LR" (logistic regression), \
        "RF" (Random Forest), "GNB"(Gaussion Naive Bayes), "SVM" (Support Vector Machine) and "DT"(Decision Tree).
    min_acc: `float`
		the minimum total accuracy to be achieved. Above this threshold, the optimization will stop.

    return
    -----
    The modified anndata object, with a slot "%s_result"%prefix
        assigned as the clustering optimization results.
    """
    # the space to use
    X = None
    if use == 'raw':
        X = ad.raw.X
    elif use == 'pca':
        if 'X_pca' not in ad.obsm.dtype.fields:
            raise ValueError("`adata.obsm['X_pca']` doesn't exist. Run `sc.pp.pca` first.")
        X = ad.obsm['X_pca']
    else:
        X = ad[:,ad.var['highly_variable']].X

    for i in range(start_iter, start_iter + n_iter):
        print("Round%d ..." % (i + 1))
        old_id = '%s_Round%d' % (prefix, i)
        new_id = '%s_Round%d' % (prefix, i + 1)

        labels = np.sort(ad.obs[old_id].unique().astype(int)).astype(str)

        # optimize
        y_prob, y_pred, y_test, clf, cvsm, acc = \
            self_projection(X, ad.obs[old_id], sparsity=sparsity, n=n,
                            fraction=fraction, classifier=classifier, n_jobs=n_jobs)
        accs = [acc]
        ad.obs['%s_self-projection' % old_id] = clf.predict(X)
        
        if plot:
            aucs = plot_roc(y_prob, y_test, clf, cvsm=cvsm, acc=acc)
            plt.show()

            sc.pl.scatter(ad, basis=basis, color=['%s_self-projection' % old_id], \
                          color_map="RdYlBu_r", legend_loc='on data', frameon=False)

        cmat = confusion_matrix(y_test, y_pred, clf, labels=labels)
        xmat = normalize_confmat1(cmat, mod)
        xmats = [xmat]
        cmats = [np.array(cmat)]
        old_id1 = old_id
        if use_projection: old_id1 = '%s_self-projection' % old_id
        for j in range(c_iter - 1):
            y_prob, y_pred, y_test, clf, _, acc = self_projection(X, ad.obs[old_id1], sparsity=sparsity, n=n,
                                                                  fraction=fraction, classifier=classifier, cv=0,
                                                                  n_jobs=n_jobs)
            accs.append(acc)
            cmat = confusion_matrix(y_test, y_pred, clf, labels=labels)
            xmat = normalize_confmat1(cmat, mod)
            xmats.append(xmat)
            cmats.append(np.array(cmat))
        R1mat = np.minimum.reduce(xmats)
        R2mat = normalize_confmat2(np.minimum.reduce(cmats))

        m1 = np.max(R1mat)
        if np.isnan(m1): m1 = 1.
        m2 = np.max(R2mat)
        print("Max R1mat: %f" % m1)
        print("Max R2mat: %f" % m2)

        if np.min(accs) > min_acc:
            ad.obs['%s_result' % prefix] = ad.obs[old_id]
            print("Converge SCCAF_optimize min_acc!")
            break
        print("min_acc: %f"%np.min(accs))

        if plot:
            if plot_cmat:
                plot_heatmap_gray(cmat, 'Confusion Matrix')
            plot_heatmap_gray(R1mat, 'Normalized Confusion Matrix (R1norm)')
            plot_heatmap_gray(R2mat, 'Normalized Confusion Matrix (R2norm)')

        if R1norm_only:
            groups = cluster_adjmat(R1mat, cutoff=R1norm_cutoff)
        elif R2norm_only:
            groups = cluster_adjmat(R2mat, cutoff=R2norm_cutoff)
        else:
            if not low_res is None:
                conn_mat = get_connection_matrix(ad_obs = ad.obs, key1 = low_res, key2 = old_id)
                zmat = np.minimum.reduce([(R1mat > R1norm_cutoff), conn_mat.values])
                groups = cluster_adjmat(zmat, cutoff=0)
            else:
                zmat = np.maximum.reduce([(R1mat > R1norm_cutoff), (R2mat > R2norm_cutoff)])
                groups = cluster_adjmat(zmat, cutoff=0)

        if len(np.unique(groups)) == len(ad.obs[old_id].unique()):
            ad.obs['%s_result' % prefix] = ad.obs[old_id]
            print("Converge SCCAF_optimize no. cluster!")
            break
        
        merge_cluster(ad, old_id1, new_id, groups)
        
        if plot:
            sc.pl.scatter(ad, basis=basis, color=[new_id], color_map="RdYlBu_r", legend_loc='on data')
        
        if len(np.unique(groups)) <= 1:
            ad.obs['%s_result' % prefix] = ad.obs[new_id]
            print("no clustering!")
            break
    
    return ad, m1, m2, np.min(accs), i


######################## PLOT FUNCTIONS ################################

def plot_link(ad, ymat, old_id, basis='tsne', ax=None, line_color='#ffa500', line_weight=10, plot_name=False,
              legend_fontsize=12):
    centroids = {}
    Y = ad.obsm['X_%s' % basis]

    for c in ad.obs[old_id].cat.categories:
        Y_mask = Y[ad.obs[old_id] == c, :]
        centroids[c] = np.median(Y_mask, axis=0)
    if plot_name:
        for c in centroids.keys():
            ax.text(centroids[c][0], centroids[c][1], c,
                    verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=legend_fontsize)

    df = ymat.copy()
    df = df.where(np.triu(np.ones(df.shape)).astype(np.bool))
    df = df.stack().reset_index()
    df.columns = ['i', 'j', 'val']
    for k in np.arange(df.shape[0]):
        val = df.iloc[k]['val']
        if df.iloc[k]['val'] > 0:
            i = df.iloc[k]['i']
            j = df.iloc[k]['j']
            ax.plot([centroids[i][0], centroids[j][0]], [centroids[i][1], centroids[j][1]],
                    linewidth=val * line_weight, color=line_color)
    return ax


def plot_center(ad, groupby, ax, basis='tsne', size=20):
    centroids = {}
    Y = ad.obsm['X_%s' % basis]

    for c in ad.obs[groupby].cat.categories:
        Y_mask = Y[ad.obs[groupby] == c, :]
        centroids[c] = np.median(Y_mask, axis=0)
    for c in centroids.keys():
        ax.plot(centroids[c][0], centroids[c][1], 'wo', markersize=size, alpha=0.5)
    return ax


def plot_link_scatter(ad, ymat, basis='pca', group='cell', title=''):
    fig, ax = plt.subplots()
    ax = plot_link(ad, ymat=ymat, old_id=group, basis=basis, ax=ax)
    sc.pl.scatter(ad, basis=basis, color=[group], color_map="RdYlBu_r", legend_loc='on data',
                  ax=ax, legend_fontsize=20, frameon=False, title=title)
    return ax


def plot_markers(top_markers, topn=10, save=None):
    """
    Plot the top marker genes as a figure.

    Input
    -----
    top_markers: `pandas dataframe`
        top weighted featured in a machine learning model.
    topn: `int`
        number of features to be plotted.
    save: `str` or None.
        Save as a figure or not. String is the save file name.

    return
    -----
    None
    """
    n_types = len(top_markers.cell_type.unique())
    nrow = int(np.floor(np.sqrt(n_types)))
    ncol = int(np.ceil(n_types / nrow))
    for i, m in enumerate(top_markers.cell_type.unique()):
        plt.subplot(nrow, ncol, i + 1)
        g = top_markers.query('cell_type == @m')
        plt.title(m, size=12, weight='bold')
        for j, gn in enumerate(g.gene):
            plt.annotate(gn, (0, 0.2 * j))
        plt.axis('off')
        plt.ylim(topn * 0.2, -0.2)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def plot_distance_jitter(df):
    ax = sns.stripplot(x="type", y="distance", data=df, jitter=True)


def sc_pl_scatter(ad, basis='tsne', color='cell'):
    df = pd.DataFrame(ad.obsm['X_%s' % basis])
    df.columns = ['%s%d' % (basis, i + 1) for i in range(df.shape[1])]
    df[color] = ad.obs[color].tolist()
    df[color] = df[color].astype('category')
    df[color].cat.categories = ad.obs[color].cat.categories
    sns.lmplot('%s1' % basis,  # Horizontal axis
               '%s2' % basis,  # Vertical axis
               data=df,  # Data source
               fit_reg=False,  # Don't fix a regression line
               hue=color,  # Set color
               scatter_kws={"marker": "o",  # Set marker style
                            "s": 10}, palette=default_20)
    return df


def plot_heatmap_gray(X, title='', save=None):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.pcolormesh(X, cmap=cm.gray_r)
    ax.set_title(title)
    # 'ax.set_xticks([])
    # 'ax.set_yticks([])
    cbaxes = fig.add_axes([1, 0.125, 0.08, 0.76])
    # 'cb = fig.colorbar(cax, cax = cbaxes, ticks=[])
    cb = fig.colorbar(cax, cax=cbaxes)
    if save:
        plt.savefig(save)
    else:
        plt.show()


def plot_roc(y_prob, y_test, clf, plot=True, save=None, title='', colors=None, cvsm=None, acc=None, fontsize=16):
    """
    y_prob, y_test, clf, plot=True, save=False, title ='', colors=None, cvsm=None, acc=None, fontsize=16):
    """
    aucs = []
    if plot:
        if colors is None:
            if len(clf.classes_) < 21:
                colors = default_20
            elif len(clf.classes_) < 27:
                colors = default_26
            else:
                colors = default_64
    for i, cell_type in enumerate(clf.classes_):
        fpr, tpr, _ = metrics.roc_curve(y_test == cell_type, y_prob[:, i])
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        if plot:
            plt.plot(fpr, tpr, c=colors[i], lw=2, label=cell_type)
    good_aucs = np.asarray(aucs)
    good_aucs = good_aucs[~np.isnan(good_aucs)]
    min_auc = np.min(good_aucs)
    max_auc = np.max(good_aucs)
    if plot:
        plt.plot([0, 1], [0, 1], color='k', ls=':')  # random
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        # 'plt.title(r'%s $AUC_{min}: %.3f \ AUC_{max}: %.3f$'%(title, min_auc,max_auc))
        plt.title(title)
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.annotate(r'$AUC_{min}: %.3f$' % min_auc, (0.5, 0.4), fontsize=fontsize)
        plt.annotate(r'$AUC_{max}: %.3f$' % max_auc, (0.5, 0.3), fontsize=fontsize)
        if cvsm:
            plt.annotate("CV: %.3f" % cvsm, (0.5, 0.2), fontsize=fontsize)
        if acc:
            plt.annotate("Test: %.3f" % acc, (0.5, 0.1), fontsize=fontsize)
        if save:
            plt.savefig(save)
    return aucs


## pySankey

class pySankeyException(Exception):
    pass


class NullsInFrame(pySankeyException):
    pass


class LabelMismatch(pySankeyException):
    pass


def check_data_matches_labels(labels, data, side):
    if len(labels) > 0:
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch('{0} labels and data do not match.{1}'.format(side, msg))


def sankey(left, right, leftWeight=None, rightWeight=None, colorDict=None,
           leftLabels=None, rightLabels=None, aspect=4, rightColor=False,
           fontsize=14, figure_name=None, closePlot=False):
    """
    Make Sankey Diagram showing flow from left-->right

    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        leftWeight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        rightWeight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding leftWeight
            is assigned
        colorDict = Dictionary of colors to use for each label
            {'label':'color'}
        leftLabels = order of the left labels in the diagram
        rightLabels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        rightColor = If true, each strip in the diagram will be be colored
                    according to its left label
    Ouput:
        None
    """
    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight

    plt.figure()
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    df = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                       'rightWeight': rightWeight}, index=range(len(left)))

    if len(df[(df.left.isnull()) | (df.right.isnull())]):
        raise NullsInFrame('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.r_[df.left.unique(), df.right.unique()]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(df.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, df['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(df.right.unique()).unique()
    else:
        check_data_matches_labels(rightLabels, df['right'], 'right')
    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        pal = "hls"
        cls = sns.color_palette(pal, len(allLabels))
        for i, l in enumerate(allLabels):
            colorDict[l] = cls[i]
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            raise RuntimeError('colorDict specified but missing values: '
                               '{}'.format(','.join(missing)))

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for l in leftLabels:
        myD_l = {}
        myD_r = {}
        for l2 in rightLabels:
            myD_l[l2] = df[(df.left == l) & (df.right == l2)].leftWeight.sum()
            myD_r[l2] = df[(df.left == l) & (df.right == l2)].rightWeight.sum()
        ns_l[l] = myD_l
        ns_r[l] = myD_r

    # Determine positions of left label patches and total widths
    widths_left = defaultdict()
    for i, l in enumerate(leftLabels):
        myD = {}
        myD['left'] = df[df.left == l].leftWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
            myD['bottom'] = widths_left[leftLabels[i - 1]]['top'] + 0.02 * df.leftWeight.sum()
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        widths_left[l] = myD

    # Determine positions of right label patches and total widths
    widths_right = defaultdict()
    for i, l in enumerate(rightLabels):
        myD = {}
        myD['right'] = df[df.right == l].rightWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
            myD['bottom'] = widths_right[rightLabels[i - 1]]['top'] + 0.02 * df.rightWeight.sum()
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        widths_right[l] = myD

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    # Draw vertical bars on left and right of each  label's section & print label
    for l in leftLabels:
        plt.fill_between(
            [-0.02 * xMax, 0],
            2 * [widths_left[l]['bottom']],
            2 * [widths_left[l]['bottom'] + widths_left[l]['left']],
            color=colorDict[l],
            alpha=0.99
        )
        plt.text(
            -0.05 * xMax,
            widths_left[l]['bottom'] + 0.5 * widths_left[l]['left'],
            l,
            {'ha': 'right', 'va': 'center'},
            fontsize=fontsize
        )
    for l in rightLabels:
        plt.fill_between(
            [xMax, 1.02 * xMax], 2 * [widths_right[l]['bottom']],
                                 2 * [widths_right[l]['bottom'] + widths_right[l]['right']],
            color=colorDict[l],
            alpha=0.99
        )
        plt.text(
            1.05 * xMax, widths_right[l]['bottom'] + 0.5 * widths_right[l]['right'],
            l,
            {'ha': 'left', 'va': 'center'},
            fontsize=fontsize
        )

    # Plot strips
    for l in leftLabels:
        for l2 in rightLabels:
            lc = l
            if rightColor:
                lc = l2
            if len(df[(df.left == l) & (df.right == l2)]) > 0:
                # Create array of y values for each strip, half at left value, half at right, convolve
                ys_d = np.array(50 * [widths_left[l]['bottom']] + 50 * [widths_right[l2]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(
                    50 * [widths_left[l]['bottom'] + ns_l[l][l2]] + 50 * [widths_right[l2]['bottom'] + ns_r[l][l2]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                # Update bottom edges at each label so next strip starts at the right place
                widths_left[l]['bottom'] += ns_l[l][l2]
                widths_right[l2]['bottom'] += ns_r[l][l2]
                plt.fill_between(
                    np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
                    color=colorDict[lc]
                )
    plt.gca().axis('off')
    #     plt.gcf().set_size_inches(6, 6)
    if figure_name != None:
        plt.savefig("{}.pdf".format(figure_name), bbox_inches='tight', dpi=150)
    if closePlot:
        plt.close()


############################ UTILS #####################################
from scipy.sparse import csr_matrix
import h5py

def readHCA(fin):
    f = h5py.File(fin, 'r')
    data = f[list(f.keys())[0]]
    X = csr_matrix((np.array(data.get("data")), 
                np.array(data.get("indices")), 
                np.array(data.get("indptr"))), 
                shape=np.array(data.get("shape"))[::-1])
    ad = sc.AnnData(X)
    ad.obs_names = np.array(data.get("barcodes")).astype(str)
    ad.var_names = np.array(data.get("gene_names")).astype(str)
    ad.var['ensembl_gene'] = np.array(data.get("genes")).astype(str)
    return(ad)
    
    
def sc_workflow(ad, prefix='L1', resolution=1.5, n_pcs=15, do_tsne=True):
    sc.pp.normalize_per_cell(ad, counts_per_cell_after=1e4)
    filter_result = sc.pp.filter_genes_dispersion(
        ad.X, min_mean=0.0125, max_mean=10, min_disp=0.25)
    sc.pl.filter_genes_dispersion(filter_result)
    print("n_HVGs: %d" % sum(filter_result.gene_subset))
    ad = ad[:, filter_result.gene_subset]
    sc.pp.log1p(ad)
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad)
    if do_tsne:
        sc.tl.tsne(ad, n_pcs=n_pcs, random_state=2)
    sc.pp.neighbors(ad, n_neighbors=10, n_pcs=n_pcs)
    sc.tl.umap(ad)
    sc.tl.louvain(ad, resolution=resolution)
    ad.obs["%s_res%.f" % (prefix, resolution)] = ad.obs["louvain"]
    return ad


def sc_pp_regress_out(adata, keys, n_jobs=None, copy=False):
    """Regress out unwanted sources of variation.
    Uses simple linear regression. This is inspired by Seurat's `regressOut`
    function in R [Satija15].
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    keys : str or list of strings
        Keys for observation annotation on which to regress on.
    n_jobs : int
        Number of jobs for parallel computation.
    copy : bool (default: False)
        If an AnnData is passed, determines whether a copy is returned.
    Returns
    -------
    Depening on `copy` returns or updates `adata` with the corrected data matrix.
    """
    logg.info('regressing out', keys, r=True)
    if issparse(adata.X):
        logg.info('... sparse input is densified and may '
                  'lead to huge memory consumption')
    adata = adata.copy() if copy else adata
    if isinstance(keys, str): keys = [keys]
    if issparse(adata.X):
        adata.X = adata.X.toarray()
    if n_jobs is not None:
        logg.warn('Parallelization is currently broke, will be restored soon. Running on 1 core.')
    n_jobs = sett.n_jobs if n_jobs is None else n_jobs

    cat_keys = []
    var_keys = []
    for key in keys:
        if key in adata.obs_keys():
            if is_categorical_dtype(adata.obs[key]):
                cat_keys.append(key)
            else:
                var_keys.append(key)
    cat_regressors = None
    if len(cat_keys) > 0:
        cat_regressors = patsy.dmatrix("+".join(cat_keys), adata.obs)
    var_regressors = None
    if len(var_keys) > 0:
        var_regressors = np.array(
            [adata.obs[key].values if key in var_keys
             else adata[:, key].X for key in var_keys]).T
    if cat_regressors is None:
        regressors = var_regressors
        if regressors is None:
            logg.warn('No correct key provided. Data not regressed out.')
            return adata
    else:
        if var_regressors is None:
            regressors = cat_regressors
        else:
            regressors = np.hstack((cat_regressors, var_regressors))

    regressors = np.c_[np.ones(adata.X.shape[0]), regressors]
    len_chunk = np.ceil(min(1000, adata.X.shape[1]) / n_jobs).astype(int)
    n_chunks = np.ceil(adata.X.shape[1] / len_chunk).astype(int)
    chunks = [np.arange(start, min(start + len_chunk, adata.X.shape[1]))
              for start in range(0, n_chunks * len_chunk, len_chunk)]

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    def _regress_out(col_index, responses, regressors):
        try:
            if regressors.shape[1] - 1 == responses.shape[1]:
                regressors_view = np.c_[regressors[:, 0], regressors[:, col_index + 1]]
            else:
                regressors_view = regressors
            result = sm.GLM(responses[:, col_index],
                            regressors_view, family=sm.families.Gaussian()).fit()
            new_column = result.resid_response
        except PerfectSeparationError:  # this emulates R's behavior
            logg.warn('Encountered PerfectSeparationError, setting to 0 as in R.')
            new_column = np.zeros(responses.shape[0])
        return new_column

    def _regress_out_chunk(chunk, responses, regressors):
        chunk_array = np.zeros((responses.shape[0], chunk.size),
                               dtype=responses.dtype)
        for i, col_index in enumerate(chunk):
            chunk_array[:, i] = _regress_out(col_index, responses, regressors)
        return chunk_array

    for chunk in chunks:
        # why did this break after migrating to dataframes?
        # result_lst = Parallel(n_jobs=n_jobs)(
        #     delayed(_regress_out)(
        #         col_index, adata.X, regressors) for col_index in chunk)
        result_lst = [_regress_out(
            col_index, adata.X, regressors) for col_index in chunk]
        for i_column, column in enumerate(chunk):
            adata.X[:, column] = result_lst[i_column]
    logg.info('finished', t=True)
    logg.hint('after `sc.pp.regress_out`, consider rescaling the adata using `sc.pp.scale`')
    return adata if copy else None


def regress_out(metadata, exprs, covariate_formula, design_formula='1', rcond=-1):
    """ Implementation of limma's removeBatchEffect function, 
    a copy from NaiveDE (https://github.com/Teichlab/NaiveDE)
    """
    # Ensure intercept is not part of covariates
    # covariate_formula is the variance to be kept, design_formula is the variance to regress out

    design_formula += ' -1'
    design_matrix = patsy.dmatrix(design_formula, metadata)

    covariate_matrix = patsy.dmatrix(covariate_formula, metadata)

    design_batch = np.hstack((covariate_matrix, design_matrix))
    coefficients, res, rank, s = np.linalg.lstsq(design_batch, exprs.T, rcond=rcond)

    beta = coefficients[covariate_matrix.shape[1]:]
    return exprs - design_matrix.dot(beta).T

