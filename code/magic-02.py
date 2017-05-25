import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import networkx as nx
from collections import defaultdict

train_orig =  pd.read_csv('../input/train.csv', header=0)
test_orig =  pd.read_csv('../input/test.csv', header=0)

df = pd.concat([train_orig, test_orig])
print("df.shape:", df.shape) # df_all.shape: (2750086, 2)

g = nx.Graph()
g.add_nodes_from(df.qid1)

edges = list(df[['qid1', 'qid2']].to_records(index=False))

g.add_edges_from(edges)
g.remove_edges_from(g.selfloop_edges())

print(len(set(df.qid1)), g.number_of_nodes()) # 4789604
print(len(df), g.number_of_edges()) # 2743365 (after self-edges)

df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])
print("df_output.shape:", df_output.shape)

NB_CORES = 20
for k in range(2, NB_CORES + 1):
    fieldname = "kcore{}".format(k)
    print("fieldname = ", fieldname)
    ck = nx.k_core(g, k=k).nodes()
    print("len(ck) = ", len(ck))
    df_output[fieldname] = 0
    df_output.ix[df_output.qid.isin(ck), fieldname] = k

    
# # "id","qid1","qid2","question1","question2","is_duplicate"
# df_id1 = train_orig[["qid1", "question1"]].drop_duplicates(keep="first").copy().reset_index(drop=True)
# df_id2 = train_orig[["qid2", "question2"]].drop_duplicates(keep="first").copy().reset_index(drop=True)

# df_id1.columns = ["qid", "question"]
# df_id2.columns = ["qid", "question"]

# print(df_id1.shape, df_id2.shape)

# df_id = pd.concat([df_id1, df_id2]).drop_duplicates(keep="first").reset_index(drop=True)
# print(df_id1.shape, df_id2.shape, df_id.shape)

# dict_questions = df_id.set_index('question').to_dict()
# dict_questions = dict_questions["qid"]

# new_id = 538000 # df_id["qid"].max() ==> 537933

# def get_id(question):
#     global dict_questions 
#     global new_id 
    
#     if question in dict_questions:
#         return dict_questions[question]
#     else:
#         new_id += 1
#         dict_questions[question] = new_id
#         return new_id
    
# rows = []
# max_lines = 10
# if True:
#     with open('../input/test.csv', 'r', encoding="utf8") as infile:
#         reader = csv.reader(infile, delimiter=",")
#         header = next(reader)
#         header.append('qid1')
#         header.append('qid2')
        
#         if True:
#             print(header)
#             pos, max_lines = 0, 10*1000*1000
#             for row in reader:
#                 # "test_id","question1","question2"
#                 question1 = row[1]
#                 question2 = row[2]

#                 qid1 = get_id(question1)
#                 qid2 = get_id(question2)
#                 row.append(qid1)
#                 row.append(qid2)

#                 pos += 1
#                 if pos >= max_lines:
#                     break
#                 rows.append(row)


# ques = pd.concat([train_orig[['question1', 'question2']], \
#         test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')

# q_dict = defaultdict(set)
# for i in range(ques.shape[0]):
#         q_dict[ques.question1[i]].add(ques.question2[i])
#         q_dict[ques.question2[i]].add(ques.question1[i])
# def q1_q2_intersect(row):
#     return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

# train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
# test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

# train_feat = train_orig[['q1_q2_intersect']]
# test_feat = test_orig[['q1_q2_intersect']]