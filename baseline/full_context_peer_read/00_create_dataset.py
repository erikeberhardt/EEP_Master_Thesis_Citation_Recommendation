#import packages
import pandas as pd
import numpy as np
import re
from scipy.sparse import data
from sklearn import preprocessing
import pickle

data_merged = pd.read_csv("df_reduced.csv")

#label encode column
le = preprocessing.LabelEncoder()
data_merged['labels'] = le.fit_transform(data_merged["target_id"].values)

references_list = data_merged.groupby("labels").count().reset_index()["labels"].to_list()
print(len(references_list))

data_merged.to_pickle("data_merged.pkl")