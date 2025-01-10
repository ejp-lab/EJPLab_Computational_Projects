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

# semi-supervised_learning_transformer.py - Generates psuedo-labels, prunes based on model confidence, and trains new model for one epoch. (DL_Pytorch)  
# To run: `python semi_supervised_learning_transformer.py [base_model]`  

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

if __name__ == "__main__":

    base_model = sys.argv[1]
    # Starting dataset is the WaltzDB
    train_csv_path = "TrainingDataset.csv"
    test_csv_path = "TestingDataset.csv"

    dataset = load_dataset('csv', data_files={'train': train_csv_path,
                                            'test': test_csv_path,})
    
    dataset = dataset.map(replace, batched=False)
    dataset = dataset.map(add_space, batched=False)
    #tokenize datasets
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    tokenized_dataset = dataset.map(lambda batch: tokenizer(batch["Sequence"]), batched=True)
    tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
    
    #split_train_dataset_obj = tokenized_dataset['train'].train_test_split(0.2, seed=42)

    # Sampling from the manifold
    manifold = pd.read_csv("PeptideManifold.csv",header=None)
    CONF_THRESH = 0.8
    len_manifold = len(manifold)
    label_to_num = {"LABEL_0":0, "LABEL_1":1}
    

    ##### ROUND 2 FIGHT #####

    model = ORTModelForSequenceClassification.from_pretrained(base_model, from_transformers=True)
    
    piplin = pipeline('text-classification',model=model,tokenizer=tokenizer, batch_size=5000, device=0)
    #sequences = sub_manifold[0].to_list()
    sequences = manifold[0].to_list()
    sequences = [" ".join(list(i)) for i in sequences]
    preds = piplin(sequences)

    # Preds are a list of dics with label
    labels = [label_to_num[i['label']] for i in preds]
    scores = [i['score'] for i in preds]

    scores_df = pd.DataFrame(np.array([sequences, labels, scores]).T, columns=['Sequence','label','Score'],)
    
    # Saving Raw Data
    scores_df.to_csv("Raw_64M_Labels_Scores.csv")

    scores_df['Score'] = scores_df['Score'].astype(np.float16)
    scores_df['label'] = scores_df['label'].astype(np.int16)
    # Removing those below threshold
    scores_df = scores_df.loc[scores_df['Score'] > CONF_THRESH]
    # Removing test set
    scores_df = scores_df.loc[~scores_df['Sequence'].isin(tokenized_dataset['test']['Sequence'])]


    new_dataset = Dataset.from_pandas(scores_df)
    new_tokenized_dataset = new_dataset.map(lambda batch: tokenizer(batch["Sequence"]), batched=True)
    new_tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
    
    m = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", num_labels=2)
    batch_size = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        "SS_Model", 
        evaluation_strategy="steps", 
        logging_strategy="steps",
        save_strategy="steps",
        seed=42, 
        learning_rate=2e-5, 
        weight_decay=0.01,
        num_train_epochs=1, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size,
        metric_for_best_model='f1',
        save_total_limit=2,
        logging_steps = 1000,
        eval_steps=10000,
        save_steps=10000,
    )

    trainer = Trainer(
        model=m,
        args=training_args,
        train_dataset=new_tokenized_dataset,
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fine,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = m(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        clf_metrics.add_batch(predictions=predictions, references=batch["labels"])

    print("metric for the whole dataset")
    metrics = clf_metrics.compute()
    print(metrics)

    pickle.dump(metrics, open("Metrics_For_Semi_Transformer.pickle",'wb'))

    trainer.save_model("Best_Semi_Supervised_Model_Retry.model")