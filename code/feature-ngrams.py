import pandas as pd
import numpy as np
from scipy.sparse import save_npz, load_npz, csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

processing = True
if processing:
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # prepare sparse
    train.loc[train.question1.isnull(), 'question1'] = ''
    train.loc[train.question2.isnull(), 'question2'] = ''
    test.loc[test.question1.isnull(), 'question1'] = ''
    test.loc[test.question2.isnull(), 'question2'] = ''

    # Bag of ngrams
    bag = CountVectorizer(
        max_df=0.999, min_df=50, max_features=30000,
        analyzer='char', ngram_range=(1, 5), binary=True, lowercase=True)

    print("Fitting")
    bag.fit(pd.concat((train.question1, train.question2)).unique())

    print("Transforming Train")
    question1_train = bag.transform(train['question1'])
    question2_train = bag.transform(train['question2'])

    print("Transforming Test")
    question1_test = bag.transform(test['question1'])
    question2_test = bag.transform(test['question2'])

    train_delta = -(question1_train != question2_train).astype(np.int8)
    test_delta = -(question1_test != question2_test).astype(np.int8)

    save_npz('../input/bag_delta_train_30k.npz', train_delta)
    save_npz('../input/bag_delta_test_30k.npz', test_delta)

read = False
if read:
    print("Reading matrices")
    q1_train = csr_matrix(load_npz('../input/q1_train.npz'), dtype=np.int32)
    q2_train = csr_matrix(load_npz('../input/q2_train.npz'), dtype=np.int32)
    q1_test = csr_matrix(load_npz('../input/q1_test.npz'), dtype=np.int32)
    q2_test = csr_matrix(load_npz('../input/q2_test.npz'), dtype=np.int32)
    print("Feature Engineering")
    bag_delta_train = csr_matrix(
        load_npz('../input/bag_delta_train.npz'), dtype=np.int8)
    bag_delta_test = csr_matrix(
        load_npz('../input/bag_delta_test.npz'), dtype=np.int8)


# cosine_sim_train = metrics.pairwise.cosine_similarity(q1_train, q2_train)



# metrics.pairwise.additive_chi2_kernel(X[, Y])
# metrics.pairwise.chi2_kernel(X[, Y, gamma])
# metrics.pairwise.distance_metrics()
# metrics.pairwise.euclidean_distances(X[, Y, ...])
# metrics.pairwise.kernel_metrics()
# metrics.pairwise.linear_kernel(X[, Y])
# metrics.pairwise.manhattan_distances(X[, Y, ...])
# metrics.pairwise.pairwise_distances(X[, Y, ...])
# metrics.pairwise.pairwise_kernels(X[, Y, ...])
# metrics.pairwise.polynomial_kernel(X[, Y, ...])
# metrics.pairwise.rbf_kernel(X[, Y, gamma])
# metrics.pairwise.sigmoid_kernel(X[, Y, ...])
# metrics.pairwise.cosine_distances(X[, Y])
# metrics.pairwise.laplacian_kernel(X[, Y, gamma])
# metrics.pairwise_distances(X[, Y, metric, ...])
# metrics.pairwise_distances_argmin(X, Y[, ...])
# metrics.pairwise_distances_argmin_min(X, Y)
# metrics.pairwise.paired_euclidean_distances(X, Y)
# metrics.pairwise.paired_manhattan_distances(X, Y)
# metrics.pairwise.paired_cosine_distances(X, Y)
# metrics.pairwise.paired_distances(X, Y[, metric])