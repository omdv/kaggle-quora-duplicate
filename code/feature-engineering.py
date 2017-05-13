#authored by https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra
from scipy.spatial.distance import euclidean, minkowski, braycurtis
from scipy.spatial.distance import chebyshev, correlation, sqeuclidean, dice
from scipy.spatial.distance import hamming, kulsinski, matching,\
  rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule
from nltk import word_tokenize

stop_words = stopwords.words('english')
np.random.seed(42)

def sent2vec(s,model):
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

# read pre-calculated Abshishek features
train_df = pd.read_csv('../input/train_features.csv', encoding = "ISO-8859-1")
test_df = pd.read_csv('../input/test_features.csv', encoding = "ISO-8859-1")

train_df['is_train'] = 1
data = pd.concat([train_df,test_df])
data = data[['question1','question2','is_train']]

# # Basic feature set - already calculated
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

# Word2Vec features
model = gensim.models.KeyedVectors.load_word2vec_format(\
  '../input/GoogleNews-vectors-negative300.bin', binary=True)
# data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)

norm_model = gensim.models.KeyedVectors.load_word2vec_format(\
  '../input/GoogleNews-vectors-negative300.bin', binary=True)
norm_model.init_sims(replace=True)
# data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0
for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q,model)

question2_vectors  = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q,model)

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

# data['chebyshev_distance'] = [chebyshev(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['correlation_distance'] = [correlation(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['sqeuclidean_distance'] = [sqeuclidean(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['hamming_distance'] = [hamming(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['kulsinski_distance'] = [kulsinski(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['matching_distance'] = [matching(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['rogerstanimoto_distance'] = [rogerstanimoto(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
# data['russellrao_distance'] = [russellrao(x, y) for (x, y) in\
#   zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]

# data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
# data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
# data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
# data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

fs4 = ['wmd','norm_wmd','cosine_distance','cityblock_distance',\
  'jaccard_distance','canberra_distance','euclidean_distance',\
  'minkowski_distance','braycurtis_distance','skew_q1vec','skew_q2vec',
  'kur_q1vec','kur_q2vec']

# train_df = data[data['is_train'].notnull()]
# test_df = data[data['is_train'].isnull()]