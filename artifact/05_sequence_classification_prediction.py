import pandas as pd
from sklearn import preprocessing
import torch
import numpy as np
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score

torch.cuda.empty_cache()

#import Torch Dataset class
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

#open tokenized X testing data
with open("X_test_tokenized.pkl", "rb") as t:
    X_test_tokenized = pickle.load(t)

#open y test data
y_true = pd.read_pickle("y_test.pkl")

# Create torch dataset
test_dataset = Dataset(X_test_tokenized) 

# Load trained model
model_path = "./output_run_6_weightdecay_fp16_epochs10_bert_large/checkpoint-15600/"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=358)

# Define test trainer
test_trainer = Trainer(model) 

# Make prediction and save them 
raw_pred, _, _ = test_trainer.predict(test_dataset) 

# Finding the indexes of the highest values of the predictions
y_pred = np.argmax(raw_pred, axis=1)

#running the raw_predictions through a softmax layer to get probabilities
softmax_predictions = tf.math.softmax(raw_pred, axis=-1)

#convert them into an array
softmax_predictions_array = softmax_predictions.numpy()

#in order to get the MRR, we need to binarize the true values (means 0 for each label, that is not correct and 1 for each label that is correct)
#"If there is exactly one relevant label per sample, label ranking average precision is equivalent to the mean reciprocal rank."
#src: https://scikit-learn.org/stable/modules/model_evaluation.html
lb = preprocessing.LabelBinarizer()
lb.fit(y_true)
y_true_binarized = lb.transform(y_true)

#print final result
print("The MRR is: " + str(label_ranking_average_precision_score(y_true_binarized, softmax_predictions_array)))