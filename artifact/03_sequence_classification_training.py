#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification
from GPUtil import showUtilization as gpu_usage
import pickle
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score


# In[13]:


# Define pretrained tokenizer and model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#have tried bert-base-uncased, will now go for distilbert-base-uncased-finetuned-sst-2-english
model_name = "bert-base-uncased"
#num_labels were 192003, now we use 358
tokenizer = BertTokenizer.from_pretrained(model_name)#, num_labels=358)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=358)


"""
#start commenting here
#No need to run this calculation as X and y tokenized have been pickled already. So they will just be imported

#here is where we decide between small and large dataset
data_merged = pd.read_pickle("data_merged_small.pkl")
print(data_merged.head(5))

#create a list so it can be trained
X = list(data_merged["sentence"])
y = list(data_merged["labels"])

# In[7]:



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


#I did it already and have pickled it so I don't have to redo the whole step all
#the time
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

with open("X_train_tokenized.pkl", "wb") as f:
    pickle.dump(X_train_tokenized, f)

with open("X_val_tokenized.pkl", "wb") as t:
    pickle.dump(X_val_tokenized, t)

with open("y_train.pkl", "wb") as t:
    pickle.dump(y_train, t)

with open("y_val.pkl", "wb") as t:
    pickle.dump(y_val, t)
     
"""
#stop comment here

with open("X_train_tokenized.pkl", "rb") as f:
    X_train_tokenized = pickle.load(f)

with open("X_val_tokenized.pkl", "rb") as t:
    X_val_tokenized = pickle.load(t)

with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("y_val.pkl", "rb") as t:
    y_val = pickle.load(t)



model = model.to(device)

# In[15]:


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# In[16]:


train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)


# In[17]:


def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    return {"accuracy": accuracy, "precision": precision, 
        "recall": recall, "f1": f1} 

# Define Trainer
args = TrainingArguments(
    output_dir="output_run_4_epochs1",
    evaluation_strategy="steps",
    #eval_steps=1300,
    #save_steps= 2600,
    do_eval=True,
    do_train=True,
    learning_rate = 2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    seed=42,
    gradient_accumulation_steps = 2,
    load_best_model_at_end=True,
    eval_accumulation_steps=700,
    #weight_decay=0.01,
    #fp16=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics    
)
print(gpu_usage())

# In[12]:
torch.cuda.empty_cache()

# Train pre-trained model
trainer.train()