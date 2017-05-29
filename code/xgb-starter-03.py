import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.cross_validation import train_test_split

np.random.seed(42)


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


def word_match_share(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))\
        / (len(q1words) + len(q2words))
    return R


def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))


def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))


def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))


def total_unq_words_stop(row, stops):
    return len(
        [x for x in set(row['question1']).union(row['question2'])
            if x not in stops])


def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))


def wc_ratio(row):
    l1 = len(row['question1']) * 1.0
    l2 = len(row['question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))


def wc_ratio_unique(row):
    l1 = len(set(row['question1'])) * 1.0
    l2 = len(set(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(
        len([x for x in set(row['question1']) if x not in stops]) -
        len([x for x in set(row['question2']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['question1']) if x not in stops]) * 1.0
    l2 = len([x for x in set(row['question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])


def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))


def char_ratio(row):
    l1 = len(''.join(row['question1']))
    l2 = len(''.join(row['question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(
        len(''.join([x for x in set(row['question1']) if x not in stops])) -
        len(''.join([x for x in set(row['question2']) if x not in stops])))


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        if word not in stops:
            q1words[word] = 1
    for word in row['question2']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights =\
        [weights.get(w, 0) for w in q1words.keys() if w in q2words] \
        + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights =\
        [weights.get(w, 0) for w in q1words] +\
        [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['question1']:
        q1words[word] = 1
    for word in row['question2']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def build_features(data, stops, weights):
    X = pd.DataFrame()
    f = functools.partial(word_match_share, stops=stops)
    X['word_match'] = data.apply(f, axis=1, raw=True)

    f = functools.partial(tfidf_word_match_share, weights=weights)
    X['tfidf_wm'] = data.apply(f, axis=1, raw=True)

    f = functools.partial(
        tfidf_word_match_share_stops, stops=stops, weights=weights)
    X['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True)

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True)
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)

    f = functools.partial(wc_diff_unique_stop, stops=stops)
    X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True)
    f = functools.partial(wc_ratio_unique_stop, stops=stops)
    X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True)

    X['same_start'] = data.apply(same_start_word, axis=1, raw=True)
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True)

    f = functools.partial(char_diff_unique_stop, stops=stops)
    X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True)

#     X['common_words'] = data.apply(common_words, axis=1, raw=True)
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)

    f = functools.partial(total_unq_words_stop, stops=stops)
    X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)

    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)

    return X


if __name__ == '__main__':
    df_train = pd.read_csv(
        '../input/train_features.csv', encoding="ISO-8859-1")
    X_train_ab = df_train.iloc[:, 2:-1]
    X_train_ab = X_train_ab.drop('euclidean_distance', axis=1)
    X_train_ab = X_train_ab.drop('jaccard_distance', axis=1)

    df_train = pd.read_csv('../input/train.csv')
    df_train = df_train.fillna(' ')

    df_test = pd.read_csv('../input/test.csv')
    ques = pd.concat(
        [df_train[['question1', 'question2']],
            df_test[['question1', 'question2']]], axis=0).\
        reset_index(drop='index')
    q_dict = defaultdict(set)
    for i in range(ques.shape[0]):
            q_dict[ques.question1[i]].add(ques.question2[i])
            q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_freq(row):
        return(len(q_dict[row['question1']]))

    def q2_freq(row):
        return(len(q_dict[row['question2']]))

    def q1_q2_intersect(row):
        return(
            len(set(q_dict[row['question1']]).
                intersection(set(q_dict[row['question2']]))))

    df_train['q1_q2_intersect'] = df_train.apply(
        q1_q2_intersect, axis=1, raw=True)
    df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
    df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

    df_test['q1_q2_intersect'] = df_test.apply(
        q1_q2_intersect, axis=1, raw=True)
    df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
    df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

    test_leaky = df_test.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]
    del df_test

    train_leaky = df_train.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]

    # explore
    stops = set(stopwords.words("english"))

    df_train['question1'] = df_train['question1'].\
        map(lambda x: str(x).lower().split())
    df_train['question2'] = df_train['question2'].\
        map(lambda x: str(x).lower().split())

    train_qs = pd.Series(
        df_train['question1'].tolist() +
        df_train['question2'].tolist())

    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    print('Building Train Features')
    X_train = build_features(df_train, stops, weights)
    X_train = pd.concat((X_train, X_train_ab, train_leaky), axis=1)
    y_train = df_train['is_duplicate'].values
    del X_train['question1']
    del X_train['question2']

    print('Building Test Features')
    df_test = pd.read_csv('../input/test_features.csv', encoding="ISO-8859-1")
    x_test_ab = df_test.iloc[:, 2:-1]
    x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
    x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)

    df_test = pd.read_csv('../input/test.csv')
    df_test = df_test.fillna(' ')

    df_test['question1'] = df_test['question1'].map(
        lambda x: str(x).lower().split())
    df_test['question2'] = df_test['question2'].map(
        lambda x: str(x).lower().split())

    x_test = build_features(df_test, stops, weights)
    X_test = pd.concat((x_test, x_test_ab, test_leaky), axis=1)
    del X_test['question1']
    del X_test['question2']

    mode = 'Val'

    if mode == 'Val':
        x_train, x_valid, y_train, y_valid =\
            train_test_split(X_train, y_train, test_size=0.2)

        # change validation to have the same distribution as test
        target = 0.175
        pos_idx = np.where(y_valid == 1)[0]
        neg_idx = np.where(y_valid == 0)[0]
        new_n_pos = np.int(target * len(neg_idx) / (1 - target))

        np.random.shuffle(pos_idx)
        idx_to_drop = np.sort(pos_idx[new_n_pos:])

        y_valid = np.delete(y_valid, idx_to_drop)
        x_valid = x_valid.drop(x_valid.index[idx_to_drop])

        # preds, model = runXGB(x_train,y_train,x_valid,y_valid,
        #           num_rounds=6000,max_depth=6,eta=0.1)

        preds, model = runLGB(
            x_train, y_train, x_valid, y_valid,
            num_rounds=10000, max_depth=6, eta=0.02, scale_pos_weight=0.36)
