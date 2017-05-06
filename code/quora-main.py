import pandas as pd
import numpy as np
import gensim
import pickle
import datetime
import xgboost as xgb
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

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)

def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

def create_submission(score, pred, model):
    """
    Saving model, features and submission
    """
    ouDir = '../output/'
    
    now = datetime.datetime.now()
    scrstr = "{:0.4f}_{}".format(score,now.strftime("%Y-%m-%d-%H%M"))
    
    mod_file = ouDir + '.model_' + scrstr + '.model'
    if model:
        print('Writing model: ', mod_file)
        pickle.dump(model,open(mod_file,'wb'))
    
    sub_file = ouDir + 'submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    pred.to_csv(sub_file, index=False)

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,\
    seed_val=0, num_rounds=2000, max_depth=6, eta=0.03):
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

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# data = pd.concat([train_df,test_df])
# data = data.drop(['id', 'qid1', 'qid2'], axis=1)

# # Basic feature set
# data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
# data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# data['diff_len'] = data.len_q1 - data.len_q2
# data['len_char_q1'] = data.question1.apply(lambda x:\
#   len(''.join(set(str(x).replace(' ', '')))))
# data['len_char_q2'] = data.question2.apply(lambda x:\
#   len(''.join(set(str(x).replace(' ', '')))))
# data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
# data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
# data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).\
#   lower().split()).intersection(set(str(x['question2']).lower().split()))),\
#   axis=1)

fs1 = ['len_q1','len_q2','diff_len','len_char_q1','len_char_q2','len_word_q1',\
  'len_word_q2','common_words']

# # FuzzyWuzzy features
# data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']),
#   str(x['question2'])), axis=1)
# data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']),
#   str(x['question2'])), axis=1)
# data['fuzz_partial_ratio'] = data.apply(lambda x:\
#   fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:\
#   fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x:\
#   fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_token_set_ratio'] = data.apply(lambda x:\
#   fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# data['fuzz_token_sort_ratio'] = data.apply(lambda x:\
#   fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

fs2 = ['fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio',\
'fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio',\
'fuzz_token_set_ratio','fuzz_token_sort_ratio']

# # Word2Vec features
# model = gensim.models.KeyedVectors.load_word2vec_format(\
#   '../input/GoogleNews-vectors-negative300.bin', binary=True)
# data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)

# norm_model = gensim.models.KeyedVectors.load_word2vec_format(\
#   '../input/GoogleNews-vectors-negative300.bin', binary=True)
# norm_model.init_sims(replace=True)
# data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

# question1_vectors = np.zeros((data.shape[0], 300))
# error_count = 0
# for i, q in tqdm(enumerate(data.question1.values)):
#     question1_vectors[i, :] = sent2vec(q)

# question2_vectors  = np.zeros((data.shape[0], 300))
# for i, q in tqdm(enumerate(data.question2.values)):
#     question2_vectors[i, :] = sent2vec(q)

# data['cosine_distance'] = [cosine(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['cityblock_distance'] = [cityblock(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['jaccard_distance'] = [jaccard(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['canberra_distance'] = [canberra(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['euclidean_distance'] = [euclidean(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]

# data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
# data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
# data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
# data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

fs4 = ['wmd','norm_wmd','cosine_distance','cityblock_distance',\
  'jaccard_distance','canberra_distance','euclidean_distance',\
  'minkowski_distance','braycurtis_distance','skew_q1vec','skew_q2vec',
  'kur_q1vec','kur_q2vec']

# read pre-calculated features
train_df = pd.read_csv('../input/train_features.csv', encoding = "ISO-8859-1")
test_df = pd.read_csv('../input/test_features.csv', encoding = "ISO-8859-1")
y = train['is_duplicate'].values

# Oversample to compensate for a different test set
pos_train = train_df[y == 1]
neg_train = train_df[y == 0]
p = 0.165
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
        ]))
    ])),
])

mode = 'Train'

if mode == 'Val':
  x_train, x_valid, y_train, y_valid =\
    train_test_split(x_train,y_train,test_size=0.2)

  x_train = pipe.fit_transform(x_train,y_train)
  x_valid = pipe.transform(x_valid)

  # model = LGBMClassifier(objective='binary',
  #   learning_rate=0.4,n_estimators=6000,max_depth=8)
  # model.fit(x_train, y_train,eval_set=[(x_valid, y_valid)],
  #   early_stopping_rounds=50)

  preds, model = runXGB(x_train,y_train,x_valid,y_valid,
            num_rounds=400,max_depth=6,eta=0.1)

if mode == 'Train':
  x_train = pipe.fit_transform(x_train,y_train)
  x_test = pipe.transform(test_df)

  # model = LGBMClassifier(objective='binary',
  #   learning_rate=0.4,n_estimators=5607,max_depth=8)
  # model.fit(x_train, y_train)

  predictions, model = runXGB(x_train,y_train,x_test,
            num_rounds=400,max_depth=6,eta=0.1)
  
  preds = pd.DataFrame()
  preds['test_id'] = test['test_id']
  preds['is_duplicate'] = predictions

  create_submission(0.320605,preds,model)