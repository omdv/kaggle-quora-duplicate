### Abhishek features
* [5886] valid_0's binary_logloss: 0.232181 - fs1,fs2,fs4 with eta=0.4, 20% split, max_depth=-1, lgbm
* [5607] valid_0's binary_logloss: 0.236258 - same but max_depth=8 (LB of 0.46953 with predict_proba[:,1])
* [399] test-mlogloss:0.320605 - trying xgboost (d=6, eta=0.1) (LB of 0.35180)
* [5999] test-logloss:0.248232 - xgboost (d=6, eta=0.1)
* [1942] valid_0's binary_error: 0.222909 - lightgbm (eta=0.1,d=6), val log_loss = 0.4253
* [4142] valid_0's binary_logloss: 0.420082 - lightgbm (eta=0.1,d=6) (LB of 0.40487)
* [4142] valid_0's binary_logloss: 0.420082 * 0.47 (to scale) (LB of 0.35372)
* [4142] valid_0's binary_logloss: 0.420082 * 0.784 (to scale to mean of public LB) (LB of 0.35789)
* [4142] valid_0's binary_logloss: 0.420082 - lightgbm (eta=0.1,d=6) with scale_pos_weight=0.36 (LB of 0.35010)
* [3218] valid_0's binary_logloss: 0.319563 / full training 0.369362 - new CV strategy, deleting pos examples to match set distribution and using scale_pos_weight (LB 0.34867)
* [1051] valid_0's binary_logloss: 0.24765 / full training 0.339638 - added freq_features (LB of 0.22529)
* [1223] valid_0's binary_logloss: 0.248323 - added log1p of len_diff and len_ratio (REVERT)
* [955]	valid_0's binary_logloss: 0.248595 - removed ratio and abs_diff and kept log1p (REVERT)
* [1051] valid_0's binary_logloss: 0.24765 - no change with eta=0.05 and max_d=-1
* [1051] valid_0's binary_logloss: 0.24765 - no change with eta=0.02 and max_d=6 (BEST LB of 0.22529 KEEP)
* [1212] valid_0's binary_logloss: 0.248568 - chebyshev (REVERT)
* [912]	valid_0's binary_logloss: 0.249056 - correlation_dist (REVERT)
* [1313] valid_0's binary_logloss: 0.248497 - sqeuclidean_distance (REVERT)
* [1308] valid_0's binary_logloss: 0.248113 - hamming_distance (REVERT)
* [1363] valid_0's binary_logloss: 0.248261 - kulsinski_distance (REVERT)
* [1308] valid_0's binary_logloss: 0.248113 - matching_distance (REVERT)
* [1536] valid_0's binary_logloss: 0.2479 - rogerstanimoto_distance (REVERT)
* [1463] valid_0's binary_logloss: 0.247302 - russellrao_distance (KEEP)
* [1727] valid_0's binary_logloss: 0.247186 / full training 0.312697- removed braycurtis_distance (KEEP, BEST CV, LB=0.22662)
* [1282] valid_0's binary_logloss: 0.248106 - removed minkowski (REVERT)
* [1513] valid_0's binary_logloss: 0.248254 - removed canberra (REVERT)
* [1315] valid_0's binary_logloss: 0.248025 - removed skew (REVERT)
* [1213] valid_0's binary_logloss: 0.245081 / full training 0.327048 - added 'q1_freq_q1_ratio','q2_freq_q1_ratio' (KEEP, BEST CV, LB=0.22209)
* [1105] valid_0's binary_logloss: 0.240065 / full training 0.318505 - added collins duffy features (KEEP)
* [2215] valid_0's binary_logloss: 0.220256 / full training 0.174624 - added 300 word2vec vectors for both questions (KEEP, BEST CV, LB=0.21ish)
* [1404] valid_0's binary_logloss: 0.239999 - best CV without word2vec after float32 conversion (REFERENCE for short case)
* [1627] valid_0's binary_logloss: 0.239735 - added 'diff_len_char','diff_len_word' (KEEP)
* [1288] valid_0's binary_logloss: 0.22262 - added top 9 starter features (KEEP)
* [2717] valid_0's binary_logloss: 0.206177 / full training 0.130557 - as above with 300 word2vec (BEST CV, BEST LB of 0.20106, LB for average of 0.19893)
* [1374] valid_0's binary_logloss: 0.227318 - added 3 gram and fixed word_hamming (REMOVE)
* [1110] valid_0's binary_logloss: 0.222018 - added countries as locations (REMOVE)
* [760]	valid_0's binary_logloss: 0.176538 - added magic feature #2 - featureq1_q2_intersect (KEEP, BEST CV, REFERENCE)
* [1265] valid_0's binary_logloss: 0.170199 / full training 0.159356 - magic #2 with word2vec (BEST CV, BEST LB 0.15505)
* [840]	valid_0's binary_logloss: 0.175802 - added qid of test (KEEP)
* [746]	valid_0's binary_logloss: 0.175691 - added qid difference (KEEP)
* [1276] valid_0's binary_logloss: 0.170703 - with word2vec and qid features (REMOVE qid)
* [755]	valid_0's binary_logloss: 0.17619 - no qid features after refactoring (REFERENCE)
* [766]	valid_0's binary_logloss: 0.155857 / full training 0.208213 - categorical encoding on qids (BEST CV, LB 0.169) (REMOVE)
* [548]	valid_0's binary_logloss: 0.156712 / full training 0.195772 - with word2vec and qid encoding (BEST CV, LB 0.165) (REMOVE)
* [713]	valid_0's binary_logloss: 0.156755 - added encoding w/o qid features (BEST CV). Only 3% on test set vs 34% on train. Need another cat feature
* [4048] valid_0's binary_logloss: 0.156085 - with X-delta of 300,000 sparse 1-10 gram
* [2828] valid_0's binary_logloss: 0.264141 - reference fs1-4, freq, starter and duffy without down-sampling
* [2000] training's binary_logloss: 0.149028 - with full X-delta and all best LB features (NO CV avail)
* [1200] training's binary_logloss: 0.180335 - with full X-delta and all best LB features (NO CV avail)
* [760]	valid_0's binary_logloss: 0.176538 - no qid after refactoring (REFERENCE)
* [669]	valid_0's binary_logloss: 0.176091 - added starter_04 (KEEP)
* [1161] valid_0's binary_logloss: 0.169939 / training's 0.165172 - previous with w2vec features (best single model LB 0.15484)

### Stacking
* [122]	valid_0's binary_logloss: 0.162682 - with deepnet, lgbm and lrg, mixed lgbm and lrg class-weight approach (lgb 1161)
* [296]	valid_0's binary_logloss: 0.168114 - same as above but without deepnet
* [94]	valid_0's binary_logloss: 0.161593 / training 0.253514 - added second version of lgbm (no class weight) and rfc (800 iters) (LB 0.14991, best LB)
* [90]	valid_0's binary_logloss: 0.159625 - added third pipeline (lgb and rfc with pipe2 w/o word2vec)
* [578]	train-logloss:0.246874	test-logloss:0.159595 - same as above but with XGB as second level clf (LB 0.14750 of two combined level 2)

### Starter-03
* [693] valid_0's binary_logloss: 0.177002 - starter 03 with lightgbm
* [2013] train-logloss:0.144105	valid-logloss:0.182807 - starter 04 with xgboost