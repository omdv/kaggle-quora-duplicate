import numpy as np
import pandas as pd

# # Merging stacknet results with listing id for submission
# pred = np.loadtxt("../stacknet/sigma_stack_pred.csv",delimiter=",")
# test = np.loadtxt("../stacknet/test_stacknet.csv",delimiter=",")
# res = np.column_stack((pred,test[:,0]))
# np.savetxt("../output/stacknet_submission.csv",\
# 	res,delimiter=",",fmt="%9.8f,%9.8f,%9.8f,%d",\
# 	header="high,medium,low,listing_id",comments='')

# Averaging with other scripts
files = [\
	"../output/submit_0.1580_2017-06-06-1854.csv",
	"../output/submit_0.1583_2017-06-04-1144.csv"
	]
weights = np.ones(len(files))/len(files)

data = []

for idx,file in enumerate(files):
	data.append(np.genfromtxt(file, dtype=float, delimiter=',', names=True))

for d in data:
	d.sort(order="test_id")

# define the result array
res = np.zeros(data[0].shape[0],\
	dtype=[('test_id','i4'),('is_duplicate','f8')])

for idx,weight in enumerate(weights):
	res["is_duplicate"] += data[idx]["is_duplicate"]*weight

res["test_id"] = data[0]["test_id"]

res = pd.DataFrame(res)

res["test_id"]=res["test_id"].astype("int")
res.to_csv("../output/submit_average_06062017-two-stacking.csv", index=False)
