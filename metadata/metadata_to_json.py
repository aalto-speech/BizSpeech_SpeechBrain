"""
Convert metadata for BizSpeech dataset from Excel Files to JSON.

Author
Anand C U
"""
import json

import pandas as pd

test_set1 = pd.read_excel(
    "test_set1.xlsx", index_col="eventId", sheet_name="Sheet1", keep_default_na=False)
test_set2 = pd.read_excel(
    "test_set2.xlsx", index_col="eventId", sheet_name="Sheet1", keep_default_na=False)
d_test1 = test_set1.to_dict(orient='index')
d_test2 = test_set2.to_dict(orient='index')
d_test2.update(d_test1)
json.dump(d_test2, open("BizSpeech_MetaData.json", "w"), sort_keys=True)

test_set1_sheets = pd.read_excel(
    "test_set1.xlsx", sheet_name=["setA", "setB", "setC"], usecols=["eventId"], squeeze=True)
test_set2_list = pd.read_excel(
    "test_set2.xlsx", sheet_name="setA", usecols=["eventId"], squeeze=True).to_list()
test_set1_list = pd.concat(
    [test_set1_sheets["setA"], test_set1_sheets["setB"], test_set1_sheets["setC"]]).to_list()
print(test_set2_list)
processed_dict = {"test_set1": [str(a) for a in test_set1_list], "test_set2": [
    str(b) for b in test_set2_list]}
for k in processed_dict:
    print(len(processed_dict[k]))
json.dump(processed_dict, open(
    "cloudasr_testset_lists.json", "w"), sort_keys=True)
