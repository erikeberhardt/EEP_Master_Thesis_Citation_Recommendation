# Citation Recommendation

The repository consists of three folders: artifact, baseline and iterations_eval.

## Artifact

The artifact folder holds the code, which describes the artifact. Scripts 01 and 02 create the respective datasets (large and small). Script 06 then fine-tunes the model, evaluates it and predicts on it. In order to run the script 06, the following needs to be done:

* Have the raw data in the same folder (sentences_with_cite.csv as well as citations.csv)
* Run scripts 01 and 02
* Adjust the first 4 parameters in script 06 to represent the correct values

## Baseline
The baseline folder is divided into two subfolders: full_context_peer_read as well as modified_full_context_peer_read. Depending on the baseline model chosen, the respective folder has to be checked. In order to run the _iteration.py file, the following steps have to be followed.

* Have the raw data in the same folder (df_reduced)
* Run script 00 to create the dataset (if working with the modified_full_context_peer_read dataset, run script 01 too)
* Adjust the first 4 parameters in script _iteration script to represent the correct values

## Iterations_eval
This folder contains the scripts used for evaluating the different iterations. The following has to be done.

* Have the raw data in the same folder (sentences_with_cite.csv as well as citations.csv)
* Run script 00 to create the dataset
* Run the respective _iteration script
