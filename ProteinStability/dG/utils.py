import os 
import re
import random 
import numpy as np 
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import joblib
import logging
import pdb

seed=42
def seed_everything(seed=seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return None

seed_everything(seed)

def add_spaces(tmp_df, seqname):

    sequences = tmp_df[seqname].to_list()
    spaced_sequences = [' '.join(list(i)) for i in sequences]
    tmp_df[seqname] = spaced_sequences

    return tmp_df

def get_pos_scalings(data, n_labels):

    label_df = pd.DataFrame(data['label'].tolist(), columns=[f'col{i+1}' for i in range(n_labels)])
    zero_to_one_ratios = label_df.apply(lambda col: col.eq(0).sum() / col.eq(1).sum() if col.eq(1).any() else np.nan).to_list()

    return torch.tensor(zero_to_one_ratios, dtype=torch.bfloat16)

def setup_trials_dir(cwd):

    trials_dir = cwd + 'Trials/'
    try:
        os.mkdir(trials_dir)
    except:
        pass

    return trials_dir

def get_datasets(in_file):

    in_df = shuffle(joblib.load(in_file))

    my_train, temp = train_test_split(in_df, test_size=0.2, random_state=seed)
    my_valid, my_test = train_test_split(temp, test_size=0.5, random_state=seed)

    my_train = shuffle(my_train)
    my_valid = shuffle(my_valid)
    my_test = shuffle(my_test)

    joblib.dump(my_train, 'TrainingSet.joblib')
    joblib.dump(my_valid, 'ValidationSet.joblib')
    joblib.dump(my_test, 'TestingSet.joblib')

    return my_train, my_valid, my_test

def loss_fn(predictions, truths):

    loss_fct = torch.nn.MSELoss()

    batch_loss = 0
    pred_list = []
    label_list = []
    
    length_batch = predictions.shape[0]

    if predictions.shape != truths.shape:
        pdb.set_trace()

    for sample_num in range(length_batch):

        sample_pred = predictions[sample_num]
        sample_truth = truths[sample_num]
        
        sample_loss = loss_fct(sample_pred, sample_truth)
        batch_loss += sample_loss
        
        sample_pred_cpu = sample_pred.cpu().detach().float().numpy().tolist()
        sample_truth_cpu = sample_truth.cpu().detach().float().numpy().tolist()

        if type(sample_pred_cpu) == float:
            pred_list.append(sample_pred_cpu)
        else:
            pred_list.extend(sample_pred_cpu)
            
        if type(sample_truth_cpu) == float:
            label_list.append(sample_truth_cpu) 
        else:
            label_list.extend(sample_truth_cpu)

    batch_loss /= length_batch
    
    return batch_loss, pred_list, label_list


def set_global_logging_level(level=logging.ERROR, prefices=['', 'torch.distributed']):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
        Default is `[""]` to match all active loggers.
        The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)