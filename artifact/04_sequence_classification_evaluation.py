import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification
from GPUtil import showUtilization as gpu_usage
import pickle

model = BertForSequenceClassification.from_pretrained('./output_run_5_weightdecay_fp16_epochs10/checkpoint-15600/', num_labels=358)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

with open("X_val_tokenized.pkl", "rb") as t:
    X_val_tokenized = pickle.load(t)

with open("y_val.pkl", "rb") as t:
    y_val = pickle.load(t)

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

test_dataset = Dataset(X_val_tokenized, y_val)


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
    output_dir="output_eval_run_1",
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    seed=42,
    eval_accumulation_steps = 700
)

test_trainer = Trainer(model=model, 
    args=args, 
    compute_metrics=compute_metrics,
    eval_dataset=test_dataset)

result = test_trainer.evaluate()

print(result)

