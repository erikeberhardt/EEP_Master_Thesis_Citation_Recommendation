#before running the script, the following parameters have to be set manually
#1/4: the model used 
#options are: "bert-large-uncased" for BERT large or "bert-base-uncased" for BERT base 
input_model = "bert-large-uncased"

#2/4: the number of labels
input_labels = 358

#input 3/4: the dataset used
#options are: "data_merged.pkl" for the large dataset and "data_merged_small.pkl" for the small dataset
input_dataset = "data_merged_small.pkl"

#input 4/4: the output directory of the trained model
#chose a path
input_outputdirectory = "./output_iteration_1/"

#import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing
from sklearn.metrics import label_ranking_average_precision_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
import torch

#set the seed to control for randomness
tf.random.set_seed(42)

#initiate the device used for training
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#define the model used for training
model_name = input_model

#initiate the tokenizer as well as the model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=input_labels)

#advise the model to use the device initiated above
model = model.to(device)

#the dataset, which is used for training, evaluating and predicting
data_merged = pd.read_pickle(input_dataset)

#based on the dataset, create a list with the values for X (sentences) and y (labels)
X = list(data_merged["sentence"])
y = list(data_merged["labels"])

#split the list into training, validation and test dataset with respective sizes of 0.8, 0.1 and 0.1
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#tokenize the sentences in all datasets
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

#create a class, which transforms the datasets into datasets, which are accepted by the model
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

#apply above function on training and validation dataset
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

#create a function which will be used to calculate standardized metrics
def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    return {"accuracy": accuracy, "precision": precision, 
        "recall": recall, "f1": f1} 

#define the arguments of the trainer
args = TrainingArguments(
    output_dir=input_outputdirectory,
    evaluation_strategy="steps",
    eval_steps=1560,
    save_steps= 1560,
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
    weight_decay=0.01,
    fp16=True
)

#define the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics    
)

#empty the cache of the gpu before initiating training
torch.cuda.empty_cache()

#train the model
trainer.train()

#evaluate the model
result = trainer.evaluate()

#rename y_test to y_true by copying it
y_true = y_test

# Create torch dataset for the sentences of the test dataset
test_dataset = Dataset(X_test_tokenized) 

# Make prediction and save them in a variable called raw_pred
raw_pred, _, _ = trainer.predict(test_dataset) 

#running the raw predictions through a softmax layer to get probabilities
softmax_predictions = tf.math.softmax(raw_pred, axis=-1)

#convert the probabilities into a numpy array
softmax_predictions_array = softmax_predictions.numpy()

#in order to get the MRR, we need to binarize the true values (means 0 for each label, that is not correct and 1 for each label that is correct)
#"If there is exactly one relevant label per sample, label ranking average precision is equivalent to the mean reciprocal rank."
#src: https://scikit-learn.org/stable/modules/model_evaluation.html

#binarize the original labels
lb = preprocessing.LabelBinarizer()
lb.fit(y)
y_true_binarized = lb.transform(y_true)

#calculate mrr
mrr = (label_ranking_average_precision_score(y_true_binarized, softmax_predictions_array))

#print mrr
print("The MRR is: " + str(mrr))