import pandas as pd
import numpy as np
import pickle
import datetime
import xgboost as xgb
import lightgbm as lgb
import tables
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix, load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

np.random.seed(42)


def convert_to_32(df):
    for i in dict(df.dtypes):
        if df.dtypes[i] == 'float64':
            df.loc[:, i] = df[i].astype(np.float32)
        if df.dtypes[i] == 'int64':
            df.loc[:, i] = df[i].astype(np.int32)
    return df


def try_apply_dict(x, dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0


class EnsembleClassifiersTransformer():
    def __init__(self, nclass=1, classifiers=None, folds=5):
        self.classifiers = classifiers
        self.kfold = StratifiedKFold(folds)
        self.nclass = nclass

    def _fit_one_fold(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def _predict_one_fold(self, X):
        res = np.ones((X.shape[0], 1)) * (-1)
        for classifier in self.classifiers:
            res = np.column_stack((res, classifier.predict_proba(X)))
        return np.array(res[:, 1:])

    def fit_transform_train(self, X, y):
        res = np.ones((X.shape[0], len(self.classifiers) * self.nclass)) * (-1)
        X_train = X
        # k-fold for training set
        for (tr_idx, cv_idx) in self.kfold.split(X_train, y):
            X_tr, y_tr = X_train[tr_idx], y[tr_idx]
            X_cv, y_cv = X_train[cv_idx], y[cv_idx]
            self._fit_one_fold(X_tr, y_tr)
            res[cv_idx, :] = self._predict_one_fold(X_cv)
            print ("Fold results (cv error):")
            for (idx, clf) in enumerate(self.classifiers):
                print(
                    "clf {:2d}: {:06.4f}".
                    format(idx, log_loss(y_cv, clf.predict_proba(X_cv))))
        return res

    def fit_transform_test(self, Xtr, ytr, Xts):
        self._fit_one_fold(Xtr, ytr)
        return self._predict_one_fold(Xts)


class CategoricalTransformer():
    def __init__(self, column_name, k=5.0, f=1.0, r_k=0.01, folds=5):
        self.k = k
        self.f = f
        self.r_k = r_k
        self.column_name = column_name
        self.folds = folds

    def _reset_fold(self):
        if hasattr(self, '_one_fold_mapping'):
            self._one_fold_mapping = {}
            self.glob_duplicate = 0

    def _fit_one_fold(self, X):
        self._reset_fold()

        tmp = X.groupby([self.column_name, 'is_duplicate']).size().\
            unstack().reset_index()
        tmp = tmp.fillna(0)

        # Get counts
        tmp['record_count'] = tmp[0] + tmp[1]
        tmp['ones_share'] = tmp[1] / tmp['record_count']
        self.glob_ones = tmp[1].sum() / tmp['record_count'].sum()

        # Get weight function
        tmp['lambda'] = 1.0 /\
            (1.0 + np.exp(np.float32(tmp['record_count'] - self.k).
             clip(-self.k, self.k) / self.f))

        # Blending
        tmp['w_ones_' + self.column_name] =\
            (1.0 - tmp['lambda']) * tmp['ones_share'] +\
            tmp['lambda'] * self.glob_ones

        # Adding random noise
        tmp['w_ones_' + self.column_name] =\
            tmp['w_ones_' + self.column_name] *\
            (1 + self.r_k * (np.random.uniform(size=len(tmp)) - 0.5))

        self._one_fold_mapping = tmp[[
            'w_ones_' + self.column_name,
            self.column_name]]
        return self

    def _transform_one_fold(self, X):
        X = pd.merge(
            X, self._one_fold_mapping, how='left', on=self.column_name)
        return X[['w_ones_' + self.column_name]]

    def fit_transform_train(self, X, y):
        kfold = StratifiedKFold(self.folds)
        res = np.ones((X.shape[0], 1)) * (-1)

        for (tr_idx, cv_idx) in kfold.split(X, y):
            self._fit_one_fold(X.iloc[tr_idx])
            tmp = self._transform_one_fold(X.iloc[cv_idx])
            res[cv_idx] = tmp.values
        tmp = pd.DataFrame(res, columns=['w_ones_' + self.column_name])
        X = pd.concat([X.reset_index(drop=True), tmp], axis=1)
        return X

    def fit_transform_test(self, Xtrain, Xtest):
        self._fit_one_fold(Xtrain)
        tmp = self._transform_one_fold(Xtest)
        Xtest = pd.concat([Xtest.reset_index(drop=True), tmp], axis=1)
        return Xtest


class PipeShape(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        print("Pipeline output shape: ", X.shape)
        return X


class PipeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.fields]


def create_submission(score, pred, model):
    """
    Saving model, features and submission
    """
    ouDir = '../output/'

    now = datetime.datetime.now()
    scrstr = "{:0.4f}_{}".format(score, now.strftime("%Y-%m-%d-%H%M"))

    mod_file = ouDir + 'model_' + scrstr + '.model'
    if model:
        print('Writing model: ', mod_file)
        pickle.dump(model, open(mod_file, 'wb'))

    sub_file = ouDir + 'submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    pred.to_csv(sub_file, index=False)


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,
           seed_val=0, num_rounds=2000, max_depth=6,
           eta=0.03, scale_pos_weight=1.0, verbose_eval=10):

    param = {}
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'logloss'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    param['scale_pos_weight'] = scale_pos_weight
    param['verbose_eval'] = 50
    param['scale_pos_weight'] = scale_pos_weight
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,
                          early_stopping_rounds=100, verbose_eval=verbose_eval)
    else:
        xgtest = xgb.DMatrix(test_X)
        watchlist = [(xgtrain, 'train')]
        model = xgb.train(plst, xgtrain, num_rounds,
                          watchlist, verbose_eval=verbose_eval)

    if test_y is None:
        pred_test_y = model.predict(xgtest)
    else:
        pred_test_y = None
    return pred_test_y, model


def runLGB(train_X, train_y, test_X=None, test_y=None, feature_names=None,
           seed_val=0, num_rounds=2000, max_depth=6,
           eta=0.03, scale_pos_weight=1.0, verbose_eval=1):

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'eta': eta,
        'max_depth': max_depth,
        'silent': 1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': seed_val,
        'verbose': 0,
        'scale_pos_weight': scale_pos_weight,
    }

    xgtrain = lgb.Dataset(train_X, train_y)

    if test_y is not None:
        xgtest = lgb.Dataset(test_X, test_y, reference=xgtrain)
        model = lgb.train(
            params, xgtrain,
            num_boost_round=num_rounds,
            valid_sets=xgtest,
            early_stopping_rounds=100,
            verbose_eval=verbose_eval)
    else:
        model = lgb.train(
            params, xgtrain,
            num_boost_round=num_rounds,
            valid_sets=xgtrain,
            verbose_eval=verbose_eval)

    if test_X is None:
        pred_test_y = None
    else:
        if test_y is None:
            pred_test_y = model.predict(test_X)
        else:
            pred_test_y = None
    return pred_test_y, model


# read original
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# read pre-calculated Abshishek features
train_df = pd.read_csv('../input/train_features.csv', encoding="ISO-8859-1")
test_df = pd.read_csv('../input/test_features.csv', encoding="ISO-8859-1")
train_df['is_duplicate'] = train['is_duplicate']
train_df.loc[:, 'is_train'] = 1
data = pd.concat([train_df, test_df])

data = convert_to_32(data)

data['diff_len_char'] = data['len_char_q1'] - data['len_char_q2']
data['diff_len_word'] = data['len_word_q1'] - data['len_word_q2']

fs1 = [
    'len_q1', 'len_q2', 'diff_len', 'len_char_q1', 'len_char_q2',
    'len_word_q1', 'len_word_q2', 'common_words',
    'diff_len_char', 'diff_len_word']

fs2 = [
    'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
    'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
    'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']

fs4 = [
    'wmd',
    'norm_wmd',
    'cosine_distance',
    'cityblock_distance',
    'jaccard_distance',
    'canberra_distance',
    'euclidean_distance',
    'minkowski_distance',
    'skew_q1vec',
    'skew_q2vec',
    'kur_q1vec',
    'kur_q2vec',
    'russellrao_distance']

# rejected features
fs5 = ['braycurtis_distance']

# split back
train_df = data[data['is_train'].notnull()]
test_df = data[data['is_train'].isnull()]
data = 0

if False:
    # -----------------------------------------------
    # Frequency features
    train_freq = pd.read_csv('../input/frequency_train.csv')
    test_freq = pd.read_csv('../input/frequency_test.csv')
    train_freq = convert_to_32(train_freq)
    test_freq = convert_to_32(test_freq)
    freq_features = [
        'q1_freq', 'q2_freq', 'q1_freq_q1_ratio',
        'q2_freq_q1_ratio', 'q1_q2_intersect']

    # -----------------------------------------------
    # XGBoost starter features - 1
    train_starter = pd.read_csv('../input/starter_train_01.csv')
    test_starter = pd.read_csv('../input/starter_test_01.csv')
    starter_features = [
        'shared_2gram', 'cosine', 'tfidf_word_match', 'word_match',
        'words_hamming', 'avg_world_len1', 'avg_world_len2',
        'stops1_ratio', 'stops2_ratio']
    train_starter = convert_to_32(train_starter)
    test_starter = convert_to_32(test_starter)

    # -----------------------------------------------
    # XGBoost starter features - 4
    train_starter_04 = pd.read_csv('../input/starter_train_04.csv')
    test_starter_04 = pd.read_csv('../input/starter_test_04.csv')
    starter_features_04 = [
        'tfidf_wm_stops', 'jaccard', 'wc_diff',
        'wc_ratio', 'wc_diff_unique', 'wc_ratio_unique', 'wc_diff_unq_stop',
        'wc_ratio_unique_stop', 'same_start', 'char_diff',
        'char_diff_unq_stop', 'total_unique_words', 'total_unq_words_stop',
        'char_ratio']
    train_starter_04 = convert_to_32(train_starter_04)
    test_starter_04 = convert_to_32(test_starter_04)

    # -----------------------------------------------
    # Collins Duffy features
    # https://www.kaggle.com/c/quora-question-pairs/discussion/32334
    train_duffy = pd.read_csv('../input/duffy_train.csv')
    test_duffy = pd.read_csv('../input/duffy_test.csv')
    duffy_features = [
        'sd_1e-1_sst', 'sd_1e-1_st', 'sd_1e-2_sst', 'sd_1e-2_st',
        'sd_1e0_sst', 'sd_1e0_st', 'sd_2e-1_sst', 'sd_2e-1_st', 'sd_5e-1_sst',
        'sd_5e-1_st', 'sd_5e-2_sst', 'sd_5e-2_st', 'sd_8e-1_sst', 'sd_8e-1_st']
    train_duffy = convert_to_32(train_duffy)
    test_duffy = convert_to_32(test_duffy)

    # -----------------------------------------------
    # Word2Vector vectors for q1 and q2
    h5 = tables.open_file('../input/w2v.h5', mode='r')
    q1_train = pd.DataFrame(np.array(h5.root.q1_train[:]).astype(np.float32))
    q2_train = pd.DataFrame(np.array(h5.root.q2_train[:]).astype(np.float32))
    q1_test = pd.DataFrame(np.array(h5.root.q1_test[:]).astype(np.float32))
    q2_test = pd.DataFrame(np.array(h5.root.q2_test[:]).astype(np.float32))
    h5.close()
    q1_train.columns = ['q1_' + str(x) for x in q1_train.columns]
    q2_train.columns = ['q2_' + str(x) for x in q2_train.columns]
    q1_test.columns = ['q1_' + str(x) for x in q1_test.columns]
    q2_test.columns = ['q2_' + str(x) for x in q2_test.columns]
    w2v_features = np.append(q1_train.columns.values, q2_train.columns.values)

# # -----------------------------------------------
# # Try question IDs
# train_df['qid1'] = train['qid1']
# train_df['qid2'] = train['qid2']
# test_df['qid1'] = test['qid1']
# test_df['qid2'] = test['qid2']
# qmax = test_df.qid2.max()
# data = pd.concat([train_df, test_df])
# data['qid1_ratio'] = data['qid1'] / qmax
# data['qid2_ratio'] = data['qid2'] / qmax
# data['qid_diff'] = data['qid1'] - data['qid2']
# train_df = data[data['is_train'].notnull()]
# test_df = data[data['is_train'].isnull()]
# data = 0
# qid_features = ['qid1', 'qid2', 'qid_diff']

# # -----------------------------------------------
# # Categorical transformer on QIDs
# cat_columns = ['qid1', 'qid2']
# cat_features = []
# for col in cat_columns:
#     cat_features.append('w_ones_' + col)

# # -----------------------------------------------
# # Bag of ngrams
# bag = CountVectorizer(
#     max_df=0.999, min_df=50, max_features=300000,
#     analyzer='char', ngram_range=(1, 10), binary=True, lowercase=True)
# train.loc[train.question1.isnull(), 'question1'] = ''
# train.loc[train.question2.isnull(), 'question2'] = ''
# print("Fitting Bag")
# bag.fit(pd.concat((train.question1, train.question2)).unique())

if False:
    # -----------------------------------------------
    # merge all
    train_df = pd.concat([train_df, train_freq], axis=1)
    test_df = pd.concat([test_df, test_freq], axis=1)

    train_freq = 0
    test_freq = 0

    train_df = pd.merge(train_df, train_duffy, on='id', how='left')
    test_df = pd.merge(test_df, test_duffy, on='id', how='left')

    train_duffy = 0
    test_duffy = 0

    train_df = pd.concat([train_df, q1_train, q2_train], axis=1)
    test_df = pd.concat([test_df, q1_test, q2_test], axis=1)

    q1_train = 0
    q2_train = 0
    q1_test = 0
    q2_test = 0

    train_df = pd.concat([train_df, train_starter], axis=1)
    test_df = pd.concat([test_df, test_starter], axis=1)

    train_starter = 0
    test_starter = 0

    train_df = pd.concat([train_df, train_starter_04], axis=1)
    test_df = pd.concat([test_df, test_starter_04], axis=1)

    train_starter_04 = 0
    test_starter_04 = 0

# split back
del train_df['is_duplicate']
train_df['is_duplicate'] = train['is_duplicate']
y_train = train['is_duplicate'].values
x_train = train_df
x_test = test_df
train_df = 0
test_df = 0

pipe = Pipeline([
    ('features', FeatureUnion([
        ('abhishek', Pipeline([
            ('get', PipeExtractor(fs1 + fs2 + fs4)),
            ('shape', PipeShape())
        ])),
        # ('frequency', Pipeline([
        #     ('get', PipeExtractor(
        #         freq_features + duffy_features +
        #         starter_features + starter_features_04)),
        #     ('shape', PipeShape())
        # ])),
        # ('w2v', Pipeline([
        #     ('get', PipeExtractor(w2v_features)),
        #     ('shape', PipeShape())
        # ])),
        # ('qid', Pipeline([
        #     ('get', PipeExtractor(qid_features)),
        #     ('shape', PipeShape())
        # ])),
        # ('categorical', Pipeline([
        #     ('get', PipeExtractor(cat_features)),
        #     ('shape', PipeShape())
        # ]))
    ])),
])

mode = 'MetaSubmit'

if mode == 'Val':
    x_test = 0
    x_train, x_valid, y_train, y_valid, idx_train, idx_valid =\
        train_test_split(x_train, y_train, x_train.index, test_size=0.2)

    # change validation to have the same distribution as test
    target = 0.175
    pos_idx = np.where(y_valid == 1)[0]
    neg_idx = np.where(y_valid == 0)[0]
    new_n_pos = np.int(target * len(neg_idx) / (1 - target))

    np.random.shuffle(pos_idx)
    idx_to_keep = np.sort(pos_idx[:new_n_pos])
    idx_to_drop = np.sort(pos_idx[new_n_pos:])

    y_valid = np.delete(y_valid, idx_to_drop)
    x_valid = x_valid.drop(x_valid.index[idx_to_drop])

    # # Categorical transformer on QIDs
    # columns = ['qid1', 'qid2']
    # for col in columns:
    #     ctf = CategoricalTransformer(col)
    #     x_train = ctf.fit_transform_train(x_train, x_train["is_duplicate"])
    #     x_valid = ctf.fit_transform_test(x_train, x_valid)

    # process pipelines
    x_train = pipe.fit_transform(x_train, y_train)
    x_valid = pipe.transform(x_valid)

    # # Bag of n-grams
    # print("Transforming Bag of features")
    # question1_train = bag.transform(train.loc[idx_train, 'question1'])
    # question2_train = bag.transform(train.loc[idx_train, 'question2'])
    # question1_valid = bag.transform(train.loc[idx_valid, 'question1'])
    # question2_valid = bag.transform(train.loc[idx_valid, 'question2'])
    # bag_delta_valid = -(question1_valid != question2_valid).astype(np.int8)
    # bag_delta_train = -(question1_train != question2_train).astype(np.int8)

    # # merge with sparse
    # x_train = hstack([csr_matrix(x_train), bag_delta_train])
    # x_valid = hstack([csr_matrix(x_valid), bag_delta_valid])
    # bag_delta_train = 0
    # bag_delta_valid = 0

    # preds, model = runLGB(
    #     x_train, y_train, x_valid, y_valid,
    #     num_rounds=10000, max_depth=6, eta=0.02, scale_pos_weight=0.36)

    x_train = np.nan_to_num(x_train)
    x_valid = np.nan_to_num(x_valid)
    lr = LogisticRegression(
        C=0.1, solver='newton-cg',
        class_weight={1: 0.472, 0: 1.309}, n_jobs=-1, max_iter=1000)
    lr.fit(x_train, y_train)
    print(log_loss(y_valid, lr.predict_proba(x_valid)[:, 1]))

elif mode == 'Train':
    # # Categorical transformer on QIDs
    # for col in cat_columns:
    #     ctf = CategoricalTransformer(col)
    #     x_train = ctf.fit_transform_train(x_train, x_train["is_duplicate"])
    #     x_test = ctf.fit_transform_test(x_train, x_test)

    # process pipeline
    x_train = pipe.fit_transform(x_train, y_train)
    x_test = pipe.transform(x_test)

    # # add sparse features
    # print("Read sparse matrix")
    # bag_delta_train = csr_matrix(
    #     load_npz('../input/bag_delta_train.npz'), dtype=np.int8)
    # print("Merge sparse matrix")
    # x_train = hstack([csr_matrix(x_train), bag_delta_train])
    # bag_delta_train = 0
    # bag_delta_test = 0

    print("Start training")

    # predictions, model = runXGB(
    #     x_train, y_train, x_test,
    #     num_rounds=6000, max_depth=6, eta=0.1)

    predictions, model = runLGB(
        x_train, y_train, x_test,
        num_rounds=1161, max_depth=6, eta=0.02, scale_pos_weight=0.36)

    # creating submission
    print("Predicting test")
    preds = pd.DataFrame()
    preds['test_id'] = test['test_id']
    preds['is_duplicate'] = predictions
    create_submission(0.169939, preds, model)

elif mode == 'TrainSparse':
    clf = MLPClassifier(
        alpha=1e-5, batch_size=256, max_iter=2,
        hidden_layer_sizes=(256, 64), random_state=1)

    # process pipeline
    x_train = pipe.fit_transform(x_train, y_train)
    x_test = pipe.transform(x_test)
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    print("Read train matrix")
    bag_delta_train = csr_matrix(
        load_npz('../input/bag_delta_train.npz'), dtype=np.int8)

    print("Merge train matrix")
    x_train = hstack([csr_matrix(x_train), bag_delta_train])
    bag_delta_train = 0

    print("Start training")
    clf.fit(x_train, y_train)
    print("Writing model to full_model.pkl")

    print("Read test matrix")
    bag_delta_test = csr_matrix(
        load_npz('../input/bag_delta_test.npz'), dtype=np.int8)

    print("Merge test matrix")
    x_test = hstack([csr_matrix(x_test), bag_delta_test])
    bag_delta_test = 0

    print("Predict test matrix")
    predictions = clf.predict_proba(x_test)
    x_test = 0

    print("Writing submission")
    preds = pd.DataFrame()
    preds['test_id'] = test['test_id']
    preds['is_duplicate'] = predictions
    create_submission(0.1000, preds, model)

elif mode == 'MetaTrain':
    rfc = RandomForestClassifier(n_estimators=800, n_jobs=-1)

    # process pipeline
    x_train = pipe.fit_transform(x_train, y_train)
    x_test = pipe.transform(x_test)

    # initiate meta array
    x_train_meta = np.ones((x_train.shape[0], 1)) * (-1)
    x_test_meta = np.ones((x_test.shape[0], 1)) * (-1)

    # predicting train out-of-fold
    kfold = StratifiedKFold(5)

    # lightgbm
    print("Start LightGBM")
    # Train
    res = np.ones((x_train.shape[0], 1)) * (-1)
    for (tr_idx, cv_idx) in kfold.split(x_train, y_train):
        preds, model = runLGB(
            x_train[tr_idx, :], y_train[tr_idx], x_train[cv_idx, :],
            num_rounds=1000, max_depth=6, eta=0.02,
            verbose_eval=100)
        res[cv_idx] = preds.reshape(-1, 1)
    x_train_meta = np.column_stack((x_train_meta, res))
    # Test
    preds, model = runLGB(
        x_train, y_train, x_test, verbose_eval=100,
        num_rounds=1000, max_depth=6, eta=0.02)
    x_test_meta = np.column_stack((x_test_meta, preds))

    # other clf section
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    for (clf, name) in [
        # (lgr, 'Logistic regression'),
        (rfc, 'Random Forest classifier')
    ]:
        print("Starting ", name)
        # Train
        res = np.ones((x_train.shape[0], 1)) * (-1)
        for (tr_idx, cv_idx) in kfold.split(x_train, y_train):
            clf.fit(x_train[tr_idx, :], y_train[tr_idx])
            preds = clf.predict_proba(x_train[cv_idx, :])[:, 1]
            print("Finished ", name, " fold...")
            res[cv_idx] = preds.reshape(-1, 1)
        x_train_meta = np.column_stack((x_train_meta, res))
        # Test
        clf.fit(x_train, y_train)
        preds = clf.predict_proba(x_test)[:, 1]
        x_test_meta = np.column_stack((x_test_meta, preds))

    # Remove dummy column
    x_train_meta = x_train_meta[:, 1:]
    x_test_meta = x_test_meta[:, 1:]

    np.savetxt("meta_train_best_single", x_train_meta)
    np.savetxt("meta_test_best_single", x_test_meta)

    # merging
    cols = ['lgb-3', 'rfc-3']
    x_train_meta = pd.DataFrame(x_train_meta)
    x_train_meta.columns = cols
    x_test_meta = pd.DataFrame(x_test_meta)
    x_test_meta.columns = cols

    x_train_new = pd.concat([
        pd.read_csv('../input/best_single_meta_train.csv'),
        x_train_meta], axis=1)
    x_test_new = pd.concat([
        pd.read_csv('../input/best_single_meta_test.csv'),
        x_test_meta], axis=1)

    x_train_new.to_csv('../input/best_single_meta_train.csv', index=False)
    x_test_new.to_csv('../input/best_single_meta_test.csv', index=False)


    # predictions, model = runLGB(
    #     x_train_meta, y_train, x_test_meta,
    #     num_rounds=100, eta=0.02, max_depth=6, scale_pos_weight=0.36)

    # # creating submission
    # preds = pd.DataFrame()
    # preds['test_id'] = test['test_id']
    # preds['is_duplicate'] = predictions
    # create_submission(0.10000, preds, model)

elif mode == 'MetaValid':
    meta_train_1 = pd.read_csv('../input/deepnet_meta_train.csv')
    meta_train_2 = pd.read_csv('../input/best_single_meta_train.csv')

    meta_train = pd.concat([
        meta_train_1,
        meta_train_2],
        axis=1)

    x_train, x_valid, y_train, y_valid =\
        train_test_split(meta_train, y_train, test_size=0.2)

    # change validation to have the same distribution as test
    target = 0.175
    pos_idx = np.where(y_valid == 1)[0]
    neg_idx = np.where(y_valid == 0)[0]
    new_n_pos = np.int(target * len(neg_idx) / (1 - target))

    np.random.shuffle(pos_idx)
    idx_to_keep = np.sort(pos_idx[:new_n_pos])
    idx_to_drop = np.sort(pos_idx[new_n_pos:])

    y_valid = np.delete(y_valid, idx_to_drop)
    x_valid = x_valid.drop(x_valid.index[idx_to_drop])

    predictions, model = runXGB(
        x_train, y_train, x_valid, y_valid,
        num_rounds=10000, max_depth=6, eta=0.02, scale_pos_weight=0.36)

elif mode == 'MetaSubmit':
    meta_train = pd.concat([
        pd.read_csv('../input/deepnet_meta_train.csv'),
        pd.read_csv('../input/best_single_meta_train.csv')],
        axis=1)

    meta_test = pd.concat([
        pd.read_csv('../input/deepnet_meta_test.csv'),
        pd.read_csv('../input/best_single_meta_test.csv')],
        axis=1)
    del meta_test['test_id']

    predictions1, model = runLGB(
        meta_train.values, y_train, meta_test.values,
        num_rounds=90, max_depth=6, eta=0.02, scale_pos_weight=0.36)
    predictions2, model = runXGB(
        meta_train.values, y_train, meta_test.values,
        num_rounds=578, max_depth=6, eta=0.02, scale_pos_weight=0.36)

    # creating submission
    preds = pd.DataFrame()
    preds['test_id'] = test['test_id']
    preds['is_duplicate'] = (predictions1 + predictions2) / 2
    # create_submission(0.159595, preds, model)
