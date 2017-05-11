import pandas as pd
import numpy as np
import gensim
import pickle
import datetime
import xgboost as xgb
import lightgbm as lgb
from fuzzywuzzy import fuzz
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard
from scipy.spatial.distance import canberra, euclidean, minkowski, braycurtis
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from lightgbm import LGBMClassifier

np.random.seed(42)
stop_words = stopwords.words('english')

def try_apply_dict(x,dict_to_apply):
    try:
        return dict_to_apply[x]
    except KeyError:
        return 0

class PipeShape(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        print("Pipeline output shape: ",X.shape)
        return X

class PipeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields
        
    def fit(self, X,y):
        return self
        
    def transform(self, X):
        return X[self.fields]

def create_submission(score, pred, model):
    """
    Saving model, features and submission
    """
    ouDir = '../output/'
    
    now = datetime.datetime.now()
    scrstr = "{:0.4f}_{}".format(score,now.strftime("%Y-%m-%d-%H%M"))
    
    mod_file = ouDir + 'model_' + scrstr + '.model'
    if model:
        print('Writing model: ', mod_file)
        pickle.dump(model,open(mod_file,'wb'))
    
    sub_file = ouDir + 'submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    pred.to_csv(sub_file, index=False)

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,\
    seed_val=0, num_rounds=2000, max_depth=6, eta=0.03, scale_pos_weight=1.0):
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
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,\
            early_stopping_rounds=100,verbose_eval=10)
    else:
        xgtest = xgb.DMatrix(test_X)
        watchlist = [ (xgtrain,'train') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,\
          verbose_eval=10)

    if test_y is None:
      pred_test_y = model.predict(xgtest)
    else:
      pred_test_y = None
    return pred_test_y, model

def runLGB(train_X, train_y, test_X, test_y=None, feature_names=None,\
    seed_val=0, num_rounds=2000, max_depth=6, eta=0.03, scale_pos_weight=1.0):
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
      'scale_pos_weight':scale_pos_weight
    }

    xgtrain = lgb.Dataset(train_X, train_y)

    if test_y is not None:
        xgtest = lgb.Dataset(test_X, test_y, reference=xgtrain)
        model = lgb.train(params, xgtrain,
          num_boost_round=num_rounds,
          valid_sets=xgtest,
          early_stopping_rounds=100)
    else:
        model = lgb.train(params, xgtrain,
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
train_df = pd.read_csv('../input/train_features.csv', encoding = "ISO-8859-1")
test_df = pd.read_csv('../input/test_features.csv', encoding = "ISO-8859-1")
train_df['is_train'] = 1
data = pd.concat([train_df,test_df])

# data['log_diff_len'] = np.log1p(data['diff_len'])
# data['ratio_len'] = data['len_q1'].apply(lambda x: x if x > 0.0 else 1.0)/\
#   data['len_q2'].apply(lambda x: x if x > 0.0 else 1.0)
# data['log_ratio_len'] = np.log1p(data['ratio_len'])

fs1 = ['len_q1','len_q2','diff_len','len_char_q1','len_char_q2','len_word_q1',\
  'len_word_q2','common_words']

fs2 = ['fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio',\
'fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio',\
'fuzz_token_set_ratio','fuzz_token_sort_ratio']

fs4 = ['wmd','norm_wmd',
  'cosine_distance',
  'cityblock_distance',\
  'jaccard_distance',
  'canberra_distance',
  'euclidean_distance',
  'minkowski_distance',
  'skew_q1vec','skew_q2vec',
  'kur_q1vec','kur_q2vec',
  'russellrao_distance']

# rejected features
fs5 = ['braycurtis_distance']

# TF-IDF features

train_df = data[data['is_train'].notnull()]
test_df = data[data['is_train'].isnull()]

# calculate frequencies
df1 = train[['question1']].copy()
df2 = train[['question2']].copy()
df1_test = test[['question1']].copy()
df2_test = test[['question2']].copy()

df2.rename(columns = {'question2':'question1'},inplace=True)
df2_test.rename(columns = {'question2':'question1'},inplace=True)

train_questions = df1.append(df2)
train_questions = train_questions.append(df1_test)
train_questions = train_questions.append(df2_test)
train_questions.drop_duplicates(subset = ['question1'],inplace=True)

train_questions.reset_index(inplace=True,drop=True)
questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()
train_cp = train.copy()
test_cp = test.copy()
train_cp.drop(['qid1','qid2'],axis=1,inplace=True)

test_cp['is_duplicate'] = -1
test_cp.rename(columns={'test_id':'id'},inplace=True)
comb = pd.concat([train_cp,test_cp])

comb['q1_hash'] = comb['question1'].map(questions_dict)
comb['q2_hash'] = comb['question2'].map(questions_dict)

q1_vc = comb.q1_hash.value_counts().to_dict()
q2_vc = comb.q2_hash.value_counts().to_dict()

#map to frequency space
comb['q1_freq'] = comb['q1_hash'].\
  map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
comb['q2_freq'] = comb['q2_hash'].\
  map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

comb['q1_freq_q1_ratio'] = comb['q1_hash'].\
  map(lambda x: try_apply_dict(x,q1_vc))
comb['q2_freq_q1_ratio'] = comb['q2_hash'].\
  map(lambda x: try_apply_dict(x,q1_vc))

comb['q1_freq_q1_ratio'] = comb['q1_freq_q1_ratio']/comb['q1_freq'].apply(lambda x: x if x > 0.0 else 1.0)
comb['q2_freq_q1_ratio'] = comb['q2_freq_q1_ratio']/comb['q2_freq'].apply(lambda x: x if x > 0.0 else 1.0)

fields = ['id','q1_hash','q2_hash','q1_freq','q2_freq','q1_freq_q1_ratio','q2_freq_q1_ratio','is_duplicate']
comb = comb[fields]

train_freq = comb[comb['is_duplicate'] >= 0]
test_freq = comb[comb['is_duplicate'] < 0]

# train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate','q1_freq_q1_ratio','q2_freq_q1_ratio']]
# test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','q1_freq_q1_ratio','q2_freq_q1_ratio']]

# merge all
train_df = pd.concat([train_df,train_freq],axis=1)
test_df = pd.concat([test_df,test_freq],axis=1)
freq_features = ['q1_freq','q2_freq','q1_freq_q1_ratio','q2_freq_q1_ratio']

y_train = train['is_duplicate'].values
x_train = train_df

# Oversample to compensate for a different distribution in test set
oversample = 0
if oversample:
  pos_train = train_df[y_train == 1]
  neg_train = train_df[y_train == 0]
  p = 0.174
  scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
  while scale > 1:
      neg_train = pd.concat([neg_train, neg_train])
      scale -=1
  neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
  x_train = pd.concat([pos_train, neg_train])
  y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
  del pos_train, neg_train

pipe = Pipeline([
    ('features', FeatureUnion([
        ('abhishek', Pipeline([
            ('get', PipeExtractor(fs1+fs2+fs4)),
            ('shape', PipeShape())
        ])),
        ('frequency', Pipeline([
            ('get', PipeExtractor(freq_features)),
            ('shape', PipeShape())
        ]))
    ])),
])

mode = 'Train'

if mode == 'Val':
  x_train, x_valid, y_train, y_valid =\
    train_test_split(x_train,y_train,test_size=0.2)

  # change validation to have the same distribution as test
  target = 0.175
  pos_idx = np.where(y_valid == 1)[0]
  neg_idx = np.where(y_valid == 0)[0]
  new_n_pos = np.int(target*len(neg_idx)/(1-target))

  np.random.shuffle(pos_idx)
  idx_to_keep = np.sort(pos_idx[:new_n_pos])
  idx_to_drop = np.sort(pos_idx[new_n_pos:])

  y_valid = np.delete(y_valid, idx_to_drop)
  x_valid = x_valid.drop(x_valid.index[idx_to_drop])

  # process pipelines  
  x_train = pipe.fit_transform(x_train,y_train)
  x_valid = pipe.transform(x_valid)

  # preds, model = runXGB(x_train,y_train,x_valid,y_valid,
  #           num_rounds=6000,max_depth=6,eta=0.1)

  preds, model = runLGB(x_train,y_train,x_valid,y_valid,
            num_rounds=10000,max_depth=6,eta=0.02,scale_pos_weight=0.36)


if mode == 'Train':
  x_train = pipe.fit_transform(x_train,y_train)
  x_test = pipe.transform(test_df)

  # predictions, model = runXGB(x_train,y_train,x_test,
  #           num_rounds=6000,max_depth=6,eta=0.1)

  predictions, model = runLGB(x_train,y_train,x_test,
            num_rounds=1213,max_depth=6,eta=0.02,scale_pos_weight=0.36)
  
  # creating submission
  preds = pd.DataFrame()
  preds['test_id'] = test['test_id']
  preds['is_duplicate'] = predictions
  create_submission(0.245081,preds,model)