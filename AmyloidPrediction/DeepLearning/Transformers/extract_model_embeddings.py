from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertForSequenceClassification
from transformers import pipeline
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import datasets
import torch
import seaborn as sns
import numpy as np
import pickle
import pdb
import polars as pl
import os
import sys

# extract_model_embeddings.py - Returns the embeddings of a model. (DL_Pytorch)  
# To run: `run extract_model_embeddings.py [model]`  

sns.set_palette('colorblind')
plt.rc('legend', fontsize=13)
#plt.rcParams["figure.figsize"] = (15,10)
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette())

def get_hidden_states(df, batch_size=100):

    def process_batch(batch):
        labels, input_ids, attention_mask = batch['label'], batch['input_ids'], batch['attention_mask']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = torch.stack(output.hidden_states).transpose(0, 1)
        hidden_states = torch.max(hidden_states, dim=1)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return hidden_states.cpu().numpy()

    dataset = datasets.Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(lambda batch: tokenizer(batch["Sequence"]), batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_hidden_states = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            hidden_states = process_batch(batch)
            all_hidden_states.append(hidden_states)

    all_hidden_states = np.concatenate(all_hidden_states, axis=0)

    return all_hidden_states

if __name__ == "__main__":

    model = sys.argv[1]

    test_df = pd.read_csv("TestingDataset.csv")
    train_df = pd.read_csv("TrainingDataset.csv")
    big_df = pl.read_csv("PeptideManifold.csv")
    big_df.columns = ["Sequence"]

    test_sequences = test_df['Sequence'].to_list()
    test_sequences = [" ".join(list(i)) for i in test_sequences]
    test_df['Sequence'] = test_sequences

    train_sequences = train_df['Sequence'].to_list()
    train_sequences = [' '.join(list(i)) for i in train_sequences]
    train_df['Sequence'] = train_sequences

    big_sequences = big_df['Sequence'].to_list()
    big_sequences = [' '.join(list(i)) for i in big_sequences]
    big_df = big_df.with_columns(pl.lit(big_sequences).alias("Sequence"))
    big_df = big_df.filter(~pl.col("Sequence").is_in(test_sequences))
    big_df = big_df.with_columns(pl.lit([0 for i in range(63999999 - len(test_sequences))]).alias("label"))
    big_df = big_df.sample(frac=0.01)
    big_df = big_df.to_pandas()

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, output_hidden_states=True )
    model = BertForSequenceClassification.from_pretrained(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_embeddings = get_hidden_states(test_df)
    train_embeddings = get_hidden_states(train_df)
    big_embeddings = get_hidden_states(big_df)

    pickle.dump(test_embeddings, open("Embedding_of_Transformer_Test.pickle", "wb"))
    pickle.dump(train_embeddings, open("Embedding_of_Transformer_Train.pickle", "wb"))
    pickle.dump(big_embeddings, open("Embedding_of_Big_Dataset.pickle", 'wb'))
