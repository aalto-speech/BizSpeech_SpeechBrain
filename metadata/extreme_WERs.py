import pymongo
import json
import numpy as np
import matplotlib.pyplot as plt
with open("/m/triton/scratch/work/umashaa1/bizspeech/bizspeech_mongoatlas.json") as f1:
    mongo_config = json.load(f1)

client = pymongo.MongoClient("mongodb+srv://" + mongo_config["username"] + ":" +
                             mongo_config["password"] + "@" + mongo_config["cluster_str"] + "/test?retryWrites=true&w=majority")
with open("exclude_event_list.json") as fh:
    exclude_list = [item for sublist in list(
        json.load(fh).values()) for item in sublist]

db = client[mongo_config["dbname"]]
collections = {"google": db.run_summary_testset1,
               "azure": db.run_summary_testset1_azure_srfixed}
set_invlid = set()
wer_invalid = []
for collection in collections.keys():
    for log in collections[collection].find({}):
        if log["average_WER"] < 0:
            if log["_id"] in exclude_list:
                print("Already known")
            else:
                print(log["_id"])
                set_invlid.add(log["_id"])
        if log["average_WER"] > 70:
            if log["_id"] in exclude_list:
                print("Already known")
            else:
                print(log["_id"], log["average_WER"])
                # wer_invalid.append(log["average_WER"])
                # set_invlid.add(log["_id"])
# density=False would make counts
n, bins, patches = plt.hist(wer_invalid, bins=10)
print(n, bins, patches)
plt.ylabel('count')
plt.xlabel('wer')
plt.show()

print([str(a) for a in list(set_invlid)])
