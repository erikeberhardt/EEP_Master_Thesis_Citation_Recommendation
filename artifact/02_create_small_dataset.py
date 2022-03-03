import pandas as pd
from sklearn import preprocessing
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

#import merged data (with labels and everything - original dataset)
data_merged = pd.read_pickle("data_merged.pkl")

#count how many sentences each citation has
data_merged_big = data_merged.groupby("labels").count().sort_values(by="article_id", ascending=False)

#only include citations, which are cited more than 100 times
big_label_list = data_merged_big.loc[data_merged_big["article_id"]>=100].reset_index()["labels"].to_list()

#new df with only these citations
data_merged_small = data_merged.loc[data_merged["labels"].isin(big_label_list)]

#reset index
data_merged_small = data_merged_small.reset_index(drop=True)

#drop original labels column
data_merged_small= data_merged_small.drop(columns={"labels"})

#encode the citations
le = preprocessing.LabelEncoder()
data_merged_small['labels'] = le.fit_transform(data_merged_small["reference_citekey"].values)

#pickle the new dataframe
data_merged_small.to_pickle("data_merged_small.pkl")