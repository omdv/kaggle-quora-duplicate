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

        tmp['record_count'] = tmp[0] + tmp[1]

        tmp['ones_share'] = tmp[1] / tmp['record_count']
        # tmp['high_share'] = tmp['high'] / tmp['record_count']
        # tmp['med_share'] = tmp['medium'] / tmp['record_count']

        # self.glob_high = tmp['high'].sum() / tmp['record_count'].sum()
        # self.glob_med = tmp['medium'].sum() / tmp['record_count'].sum()
        self.glob_ones = tmp[1].sum() / tmp['record_count'].sum()

        # Get weight function
        tmp['lambda'] = 1.0 /\
            (1.0 + np.exp(np.float32(tmp['record_count'] - self.k).
             clip(-self.k, self.k) / self.f))

        # Blending
        # tmp['w_high_'+self.column_name] =\
        #     (1.0-tmp['lambda'])*tmp['high_share']+tmp['lambda']*self.glob_high
        # tmp['w_med_'+self.column_name] =\
        #     (1.0-tmp['lambda'])*tmp['med_share']+tmp['lambda']*self.glob_med
        tmp['w_ones_' + self.column_name] =\
            (1.0 - tmp['lambda']) * tmp['ones_share'] +\
            tmp['lambda'] * self.glob_ones

        # Adding random noise
        # tmp['w_high_' + self.column_name] =\
        #   tmp['w_high_' + self.column_name]*\
        #     (1+self.r_k*(np.random.uniform(size = len(tmp))-0.5))
        # tmp['w_med_' + self.column_name] = tmp['w_med_' + self.column_name]*\
        #     (1+self.r_k*(np.random.uniform(size = len(tmp))-0.5))

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
           eta=0.03, scale_pos_weight=1.0):

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
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,
                          early_stopping_rounds=100, verbose_eval=10)
    else:
        xgtest = xgb.DMatrix(test_X)
        watchlist = [(xgtrain, 'train')]
        model = xgb.train(plst, xgtrain, num_rounds,
                          watchlist, verbose_eval=10)

    if test_y is None:
        pred_test_y = model.predict(xgtest)
    else:
        pred_test_y = None
    return pred_test_y, model


def runLGB(train_X, train_y, test_X, test_y=None, feature_names=None,
           seed_val=0, num_rounds=2000, max_depth=6,
           eta=0.03, scale_pos_weight=1.0):

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
        'scale_pos_weight': scale_pos_weight
    }

    xgtrain = lgb.Dataset(train_X, train_y)

    if test_y is not None:
        xgtest = lgb.Dataset(test_X, test_y, reference=xgtrain)
        model = lgb.train(
            params, xgtrain,
            num_boost_round=num_rounds,
            valid_sets=xgtest,
            early_stopping_rounds=100)
    else:
        model = lgb.train(
            params, xgtrain,
            num_boost_round=num_rounds,
            valid_sets=xgtrain)

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

# data['log_diff_len'] = np.log1p(data['diff_len'])
# data['ratio_len'] = data['len_q1'].apply(lambda x: x if x > 0.0 else 1.0)/\
#   data['len_q2'].apply(lambda x: x if x > 0.0 else 1.0)
# data['log_ratio_len'] = np.log1p(data['ratio_len'])

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

# # -----------------------------------------------
# # calculate frequencies
# df1 = train[['question1']].copy()
# df2 = train[['question2']].copy()
# df1_test = test[['question1']].copy()
# df2_test = test[['question2']].copy()

# df2.rename(columns = {'question2':'question1'},inplace=True)
# df2_test.rename(columns = {'question2':'question1'},inplace=True)

# train_questions = df1.append(df2)
# train_questions = train_questions.append(df1_test)
# train_questions = train_questions.append(df2_test)

# train_questions.drop_duplicates(subset = ['question1'],inplace=True)

# train_questions.reset_index(inplace=True,drop=True)
# questions_dict = pd.Series(train_questions.index.values,index=\
#  train_questions.question1.values).to_dict()
# train_cp = train.copy()
# test_cp = test.copy()
# train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

# test_cp['is_duplicate'] = -1
# test_cp.rename(columns={'test_id':'id'},inplace=True)
# comb = pd.concat([train_cp,test_cp])

# comb['q1_hash'] = comb['question1'].map(questions_dict)
# comb['q2_hash'] = comb['question2'].map(questions_dict)

# q1_vc = comb.q1_hash.value_counts().to_dict()
# q2_vc = comb.q2_hash.value_counts().to_dict()

# #map to frequency space
# comb['q1_freq'] = comb['q1_hash'].\
#  map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
# comb['q2_freq'] = comb['q2_hash'].\
#  map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

# comb['q1_freq_q1_ratio'] = comb['q1_hash'].\
#  map(lambda x: try_apply_dict(x,q1_vc))
# comb['q2_freq_q1_ratio'] = comb['q2_hash'].\
#  map(lambda x: try_apply_dict(x,q1_vc))

# comb['q1_freq_q1_ratio'] = comb['q1_freq_q1_ratio']/\
#  comb['q1_freq'].apply(lambda x: x if x > 0.0 else 1.0)
# comb['q2_freq_q1_ratio'] = comb['q2_freq_q1_ratio']/\
#  comb['q2_freq'].apply(lambda x: x if x > 0.0 else 1.0)

# fields = ['id','q1_hash','q2_hash','q1_freq','q2_freq',\
#  'q1_freq_q1_ratio','q2_freq_q1_ratio','is_duplicate']
# comb = comb[fields]

# train_freq = comb[comb['is_duplicate'] >= 0]
# test_freq = comb[comb['is_duplicate'] < 0]

train_freq = pd.read_csv('../input/frequency_train.csv')
test_freq = pd.read_csv('../input/frequency_test.csv')

train_freq = convert_to_32(train_freq)
test_freq = convert_to_32(test_freq)

freq_features = [
    'q1_freq', 'q2_freq', 'q1_freq_q1_ratio',
    'q2_freq_q1_ratio', 'q1_q2_intersect']

# -----------------------------------------------
# XGBoost start features
train_starter = pd.read_csv('../input/starter_train_01.csv')
test_starter = pd.read_csv('../input/starter_test_01.csv')

starter_features = [
    'shared_2gram', 'cosine', 'tfidf_word_match', 'word_match',
    'words_hamming', 'avg_world_len1', 'avg_world_len2',
    'stops1_ratio', 'stops2_ratio']

train_starter = convert_to_32(train_starter)
test_starter = convert_to_32(test_starter)

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
# # Explore other frequency features
# def get_weight(count, eps=10000, min_count=2):
#   return 0 if count < min_count else 1 / (count + eps)

# all_qs = np.concatenate((train.question1.values,train.question2.values,\
#   test.question1.values,test.question2.values))
# all_qs = [str(i).lower() for i in all_qs]

# qs_counts = Counter(all_qs)

# words = (" ".join(train_qs)).lower().split()
# word_counts = Counter(words)
# word_weights = {word: get_weight(count) for word, count\
# in word_counts.items()}

# -----------------------------------------------
# Try question IDs
train_df['qid1'] = train['qid1']
train_df['qid2'] = train['qid2']
test_df['qid1'] = test['qid1']
test_df['qid2'] = test['qid2']

qmax = test_df.qid2.max()

data = pd.concat([train_df, test_df])
data['qid1_ratio'] = data['qid1'] / qmax
data['qid2_ratio'] = data['qid2'] / qmax
data['qid_diff'] = data['qid1'] - data['qid2']

train_df = data[data['is_train'].notnull()]
test_df = data[data['is_train'].isnull()]
data = 0

qid_features = ['qid1', 'qid2', 'qid_diff']

# -----------------------------------------------
# Categorical transformer on QIDs
columns = ['qid1', 'qid2']
cat_features = []
for col in columns:
    ctf = CategoricalTransformer(col)
    train_df = ctf.fit_transform_train(train_df, train_df["is_duplicate"])
    test_df = ctf.fit_transform_test(train_df, test_df)
    cat_features.append('w_ones_' + col)

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

# split back
y_train = train['is_duplicate'].values
x_train = train_df

pipe = Pipeline([
    ('features', FeatureUnion([
        ('abhishek', Pipeline([
            ('get', PipeExtractor(fs1 + fs2 + fs4)),
            ('shape', PipeShape())
        ])),
        ('frequency', Pipeline([
            ('get', PipeExtractor(freq_features)),
            ('shape', PipeShape())
        ])),
        ('collins_duffy', Pipeline([
            ('get', PipeExtractor(duffy_features)),
            ('shape', PipeShape())
        ])),
        ('starter', Pipeline([
            ('get', PipeExtractor(starter_features)),
            ('shape', PipeShape())
        ])),
        # ('qid', Pipeline([
        #     ('get', PipeExtractor(qid_features)),
        #     ('shape', PipeShape())
        # ])),
        ('categorical', Pipeline([
            ('get', PipeExtractor(cat_features)),
            ('shape', PipeShape())
        ])),
        ('w2v', Pipeline([
            ('get', PipeExtractor(w2v_features)),
            ('shape', PipeShape())
        ]))
    ])),
])

mode = 'Train'

if mode == 'Val':
    x_train, x_valid, y_train, y_valid =\
        train_test_split(x_train, y_train, test_size=0.2)

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

    # process pipelines
    x_train = pipe.fit_transform(x_train, y_train)
    x_valid = pipe.transform(x_valid)

    # preds, model = runXGB(x_train,y_train,x_valid,y_valid,
    #           num_rounds=6000,max_depth=6,eta=0.1)

    preds, model = runLGB(
        x_train, y_train, x_valid, y_valid,
        num_rounds=10000, max_depth=6, eta=0.02, scale_pos_weight=0.36)

if mode == 'Train':
    x_train = pipe.fit_transform(x_train, y_train)
    x_test = pipe.transform(test_df)

    # predictions, model = runXGB(x_train,y_train,x_test,
    #           num_rounds=6000,max_depth=6,eta=0.1)

    predictions, model = runLGB(
        x_train, y_train, x_test,
        num_rounds=548, max_depth=6, eta=0.02, scale_pos_weight=0.36)

    # creating submission
    preds = pd.DataFrame()
    preds['test_id'] = test['test_id']
    preds['is_duplicate'] = predictions
    create_submission(0.156712, preds, model)
