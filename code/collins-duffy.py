import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import os

df_train = None
flip = True
base = '../input/collins_duffy_train/'
count = 0
for filename in os.listdir(base):
    if filename.endswith(".csv"): 
        tmpFrame = pd.read_csv(os.path.join(os.getcwd(), base, filename))
        print(os.path.join(os.getcwd(), base, filename))
        tmpFrame = tmpFrame.rename(columns={'cdNorm_st': filename.replace(".csv","")})
        if flip:
            df_train = tmpFrame
            flip = False
        else:
            df_train = df_train.merge(tmpFrame,how='inner',on='id')
        count+=1
        continue
    else:
        continue

df_test = None
flip = True
base = '../input/collins_duffy_test/'
count = 0
for filename in os.listdir(base):
    if filename.endswith(".csv"): 
        tmpFrame = pd.read_csv(os.path.join(os.getcwd(), base, filename))
        print(os.path.join(os.getcwd(), base, filename))
        tmpFrame = tmpFrame.rename(columns={'cdNorm_st': filename.replace(".csv","")})
        if flip:
            df_test = tmpFrame
            flip = False
        else:
            df_test = df_test.merge(tmpFrame,how='inner',on='id')
        count+=1
        continue
    else:
        continue