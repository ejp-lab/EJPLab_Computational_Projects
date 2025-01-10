from transformers import BertTokenizer
import re
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, IntervalStrategy, BertForSequenceClassification
from datasets import load_metric
from transformers import EarlyStoppingCallback
import numpy as np
import evaluate
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import random
import pandas as pd
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline
from time import time
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
import pdb
import optuna
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import sys

# prospective_predictions.py - Creates preditictions from the semi-supervised transformer for the 64M peptides. (DL_Pytorch)  
# To run: `python prospective_predictions.py [model]`  

os.environ["WANDB_DISABLED"] = "true"
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED']=str(42)
SEED =42

def replace(x):
    x["Sequence"] = re.sub(r"[UZOB]", "X", x["Sequence"].upper())
    return x

def add_space(x):
    x["Sequence"] = ' '.join(x["Sequence"])
    return x

def compute_metrics_fine(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = balanced_accuracy_score(y_true=labels, y_pred=predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions,average='binary')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

model = sys.argv[1]
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )

model = ORTModelForSequenceClassification.from_pretrained(model, from_transformers=True)
manifold = pd.read_csv("PeptideManifold.csv",header=None)
CONF_THRESH = 0.8
len_manifold = len(manifold)
label_to_num = {"LABEL_0":0, "LABEL_1":1}

piplin = pipeline('text-classification',model=model,tokenizer=tokenizer, batch_size=5000, device=0)
sequences = manifold[0].to_list()
sequences = [" ".join(list(i)) for i in sequences]
preds = piplin(sequences)

labels = [label_to_num[i['label']] for i in preds]
scores = [i['score'] for i in preds]

scores_df = pd.DataFrame(np.array([sequences, labels, scores]).T, columns=['Sequence','label','Score'],)

# Saving Raw Data
scores_df.to_csv("Prospective_Predictions_64M.csv")
