import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from tqdm import tqdm

from subprocess import check_output

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
locations = pd.read_csv("../input/cities.csv")

df = pd.concat([train,test])

# There's lots of room to add more locations, but start with just countries
countries = set(locations['Country'].dropna(inplace=False).values.tolist())
cities = set(locations['City'].dropna(inplace=False).values.tolist())
all_places = countries

# Turn it into a Regex
regex = "|".join(sorted(set(all_places)))

results = []
print("processing:", df.shape)
for index, row in tqdm(df[0:].iterrows()):
    q1 = str(row['question1'])
    q2 = str(row['question2'])

    rr = {}

    q1_matches = []
    q2_matches = []

    if (len(q1) > 0):
        q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]

    if (len(q2) > 0):
        q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]

    rr['z_q1_place_num'] = len(q1_matches)
    rr['z_q1_has_place'] = len(q1_matches) > 0

    rr['z_q2_place_num'] = len(q2_matches) 
    rr['z_q2_has_place'] = len(q2_matches) > 0

    rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))
    rr['z_place_match'] = rr['z_place_match_num'] > 0

    rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))
    rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0

    results.append(rr)     

out_df = pd.DataFrame.from_dict(results)
out_train = out_df.loc[0:404289,:]
out_test = out_df.loc[404290:,:]

out_train.to_csv('../input/location_train.csv',index=False)
out_test.to_csv('../input/location_test.csv',index=False)