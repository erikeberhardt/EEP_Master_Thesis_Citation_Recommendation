#import packages
import pandas as pd
import re
from sklearn import preprocessing

#import raw data
sentences_with_cite = pd.read_csv("sentences_with_cite.csv")
citations = pd.read_csv("citations.csv")

#filter the raw data, so we are only looking at sentences, which have a citation
#this is needed as the training of the model requires sentence-citation pairs
sub_sentences_with_cite = sentences_with_cite.loc[sentences_with_cite["has_citation"] == 1]

#create subset so training is faster
sub_sentences_with_cite = sub_sentences_with_cite.sample(n=60_000, random_state=42)

#the following loop removes punctuation symbols from the term "et al"
#afterwards, everything is put into a data frame
cleaned_output = []
for i in range(len(sub_sentences_with_cite)):
    x = sub_sentences_with_cite.iloc[i][10]
    newli = []
    if "et al ." in x or "et al." in x:
        test_al_no_dot = x
        test_al_no_dot = str(test_al_no_dot).replace("et al .", "et al ")
        test_al_no_dot = str(test_al_no_dot).replace("et al.", "et al ")
        newli.append(sub_sentences_with_cite.iloc[i][1])
        newli.append(sub_sentences_with_cite.iloc[i][2])
        newli.append(test_al_no_dot)
        cleaned_output.append(newli)
    else:
        newli.append(sub_sentences_with_cite.iloc[i][1])
        newli.append(sub_sentences_with_cite.iloc[i][2])
        newli.append(x)
        cleaned_output.append(newli)
df = pd.DataFrame(cleaned_output,columns=["article_id", "sentence_id", "sentence"])

#a regular expression operation function is created, which removes citations and whitespaces
def regex(input):
    output = re.sub('\s+CITE.[^.!?   ]*', '', str(input), flags=re.I) #remove citations
    output = re.sub(r'\s([?.!"](?:\s|$))', r'\1', output) #remove whitespaces
    return output

#the above defined function is applied on the dataset on the sentence column
df["sentence"] = df["sentence"].apply(regex)

#merge the processed sentences to the respective citations on article- and sentence_id
data_merged = pd.merge(df, citations, on=["article_id", "sentence_id"])
data_merged = data_merged[{"article_id", "sentence_id", "sentence", "reference_citekey"}]

#the citations are encoded so they can be processed by the model
le = preprocessing.LabelEncoder() #initialize the encoder
data_merged['labels'] = le.fit_transform(data_merged["reference_citekey"].values) #store the encoded citations in a column named labels

#print anumber of labes
print(len(data_merged.groupby("labels").count()))

#pickle the data so this process does not have to be repeated all the time
data_merged.to_pickle("data_merged.pkl")