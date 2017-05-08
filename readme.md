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
* [1051] valid_0's binary_logloss: 0.24765 / full training 0.339638 - added freq_features (BEST LB of 0.22529)