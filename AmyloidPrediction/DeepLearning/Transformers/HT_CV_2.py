from transformers import BertModel, BertTokenizer
import re
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, IntervalStrategy
from datasets import load_metric
import numpy as np
import evaluate
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import random
import optuna
from optuna.samplers import TPESampler
from transformers import BertForSequenceClassification
import pickle
from datetime import datetime

# HT_CV_2.py - Contains the 5-fold cross validation for the ProteinBert transformer. (DL_Pytorch)  
# To run: `python HT_CV_2.py`  
os.environ["WANDB_DISABLED"] = "true"
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED']=str(42)

def replace(x):
    x["Sequence"] = re.sub(r"[UZOB]", "X", x["Sequence"].upper())
    return x
def add_space(x):
    x["Sequence"] = ' '.join(x["Sequence"])
    return x


def objective (trial):
    
    #parameters to search
    params = {
    
    'weight_decay': trial.suggest_float('weight_decay', 0, 5e-3, step=5e-4),
    'batch_size' : trial.suggest_int('batch_size', 8, 128, step=8),
    'initial_learning_rate' : trial.suggest_float('initial_learning_rate', 1e-6, 3e-5, log=True),    
    'epochs' : trial.suggest_int('epochs', 5, 30, step=1),
    }

    prediction_total = []
    truth_total = []
    for train_index, val_index in kf.split(seq_train, label_train):
        
        m = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", num_labels=2)

        # splitting Dataframe (dataset not included)
        train_part = tokenized_dataset['train'][train_index]
        val_part = tokenized_dataset['train'][val_index]
        train_dataset = Dataset.from_dict(train_part)
        eval_dataset = Dataset.from_dict(val_part)
        #for eval
        eval_dataloader = DataLoader(
        eval_dataset, batch_size=32, collate_fn=data_collator
        )

        #trainer arguments
        training_args = TrainingArguments(
        output_dir="./training_results",
        learning_rate=params["initial_learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["epochs"],
        weight_decay=params["weight_decay"],
        label_names=["label"],#"Class"
        logging_steps=200,
        save_strategy = 'no',
        load_best_model_at_end=False,
        seed = 42,
        save_total_limit = 2,
        metric_for_best_model = "f1"
        )

        #train the model
        trainer = Trainer(
            model=m,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=clf_metrics
            
        )
        trainer.train()

        prediction_holder = []
        truth_holder = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = m(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            clf_metrics.add_batch(predictions=predictions, references=batch["labels"])
            #append predictions and truth
            prediction_holder.append(predictions.tolist())
            truth_holder.append((batch["labels"]).tolist())
        
        #print out evaluating metrics for every fold
        print('metric for skfold')
        print(clf_metrics.compute())

        prediction_total += prediction_holder
        truth_total += truth_holder

    #Combined CV evaluation 
    prediction_flat = [num for sublist in prediction_total for num in sublist]
    truth_flat = [num for sublist in truth_total for num in sublist]
    clf_metrics.add_batch(predictions=prediction_flat, references=truth_flat)
    print("Full CV metrics")
    x = clf_metrics.compute()
    print('The metrics for full CV')
    print(x)
    
    return x['f1']

    
if __name__ == "__main__":

    #set tokenizer and the model
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    
    #Prtepatre data
    train_csv_path = "Clustered_Train_Set.csv"
    test_csv_path = "Clustered_Test_Set.csv"

    dataset = load_dataset('csv', data_files={'train': train_csv_path,
                                            'test': test_csv_path,})
    dataset = dataset.map(replace, batched=False)
    dataset = dataset.map(add_space, batched=False)

    #tokenize datasets
    tokenized_dataset = dataset.map(lambda batch: tokenizer(batch["Sequence"]), batched=True)
    tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])

    #define metrics
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Sk-fold set-up
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    seq_train = tokenized_dataset['train']['Sequence']
    label_train = tokenized_dataset['train']['label']
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    ##############################
    ########## OPTUNA ############
    ##############################

    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed = 42))
    study.optimize(objective, n_trials=25, n_jobs=1, show_progress_bar=True)
    
    # BEST Parameters
    best_params = study.best_trial.params
    print("The best parameter is: ")
    print(best_params)

    ##############################
    ###### Test on Testset #######
    ##############################
    
    #construct the model
    m = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./training_results",
        learning_rate=best_params['initial_learning_rate'],
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        weight_decay=best_params["weight_decay"],
        label_names=["label"],#"Class"
        logging_steps=200,
        save_strategy = 'no',
        load_best_model_at_end=False,
        seed = 42,
        save_total_limit = 2,
        num_train_epochs=best_params['epochs']
    )

    #Train the whole datasets
    split_train_dataset_obj = tokenized_dataset['train']
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=32, collate_fn=data_collator
    )

    trainer = Trainer(
            model=m,
            args=training_args,
            train_dataset=split_train_dataset_obj,
            data_collator=data_collator,
            compute_metrics=clf_metrics
    )
    trainer.train()
    trainer.save_model("best_model_trained_on_whole_datasets.model")

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = m(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        clf_metrics.add_batch(predictions=predictions, references=batch["labels"])
        

    print("metric for the whole dataset on test dataset")
    metrics = clf_metrics.compute()
    print(metrics)
    pickle.dump(metrics, open("Metrics_For_CV_Transformer.pickle",'wb'))
    pickle.dump(best_params, open("Best_CV_Transformer_Params.pickle", 'wb'))
    pickle.dump(study, open(f"Optuna_Study_CV_5_{datetime()}.pickle",'wb'))
