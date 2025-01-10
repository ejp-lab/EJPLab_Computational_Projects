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
from scipy.interpolate import interp1d
import sys

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
def find_bin_center(z):
    arr=[]
    for i in range(0,len(z)-1):
        arr.append((z[i]+z[i+1])/2)
    
    return arr

def PlotHistogram(labs, probs, title):

    fig, axes = plt.subplots()


    zero_inds = [idx for idx, i in enumerate(labs) if i == 0]
    one_inds = [idx for idx, i in enumerate(labs) if i == 1]

    zero_probs = [probs[i] for i in zero_inds]
    one_probs = [probs[i] for i in one_inds]

    lrprobs_non = np.array(zero_probs)
    lrprobs_hits = np.array(one_probs)

    binwidth = 0.025

    weights = np.ones_like(lrprobs_non) / float(len(lrprobs_non))
    non = plt.hist(lrprobs_non, weights=weights, bins=np.arange(0, 1 + binwidth, binwidth), color='blue', alpha=0.45)
    non_bins = non[1]
    non = non[0]

    weights = np.ones_like(lrprobs_hits) / float(len(lrprobs_hits))
    hits = plt.hist(lrprobs_hits, weights=weights, bins=np.arange(0, 1 + binwidth, binwidth), color='red', alpha=0.45)
    hits_bins = hits[1]
    hits = hits[0]

    x = np.array(find_bin_center(non_bins))
    y = np.array(non)

    max_y = max(y)

    x_new = np.linspace(x.min(), x.max(), 500)

    f = interp1d(x, y, kind='quadratic')
    y_smooth = f(x_new)

    plt.plot (x_new, y_smooth, color='blue', alpha=0.45)

    x = np.array(find_bin_center(hits_bins))
    y = np.array(hits)
    x_new = np.linspace(x.min(), x.max(), 500)

    f = interp1d(x, y, kind='quadratic')
    y_smooth = f(x_new)

    plt.plot(x_new, y_smooth, color='red', alpha=0.45)

    if max(y) > max_y:
        max_y = max(y)
    else:
        pass

    plt.vlines(0.5, ymin=0, ymax=max_y + 0.025, color='black', linestyles='dashed')

    plt.ylim(0, max_y + 0.025)

    plt.xlabel('Probability of Being Amyloid Forming')
    plt.ylabel('Normalized Frequency')
    axes.minorticks_on()
    plt.xlim(0, 1)
    fig.tight_layout()
    plt.savefig(title)
    plt.close()

    return True

if __name__ == "__main__":

    model_1 = sys.argv[1]
    model_2 = sys.argv[2]
    hist_name_1 = sys.argv[3]
    hist_name_2 = sys.argv[4]
    roc_plot_name = sys.argv[5]

    test_df = pd.read_csv("TestingDataset.csv")
    y_true = test_df['label'].to_list()
    sequences = test_df['Sequence'].to_list()
    sequences = [" ".join(list(i)) for i in sequences]
    test_df['Sequence'] = sequences

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, output_hidden_states=True )
    model = BertForSequenceClassification.from_pretrained(model_1)

    dataset = datasets.Dataset.from_pandas(test_df)
    tokenized_dataset = dataset.map(lambda batch: tokenizer(batch["Sequence"]), batched=True)
    tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(tokenized_dataset, batch_size=1000, shuffle=False)

    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in dataloader:
            labels, input_ids, attention_mask = batch['label'], batch['input_ids'], batch['attention_mask']
            output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = output.logits
            predictions.append(logits.softmax(dim=1).cpu().numpy())
    
    y_score = predictions[0][:,1]

    # Calculate TPR and FPR
    fpr_init, tpr_init, thresholds_init = roc_curve(y_true, y_score)

    # Calculate AUC
    roc_auc_init = auc(fpr_init, tpr_init)

    # Classification histogram
    PlotHistogram(y_true, y_score, hist_name_1)

    model = BertForSequenceClassification.from_pretrained(model_2)

    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in dataloader:
            labels, input_ids, attention_mask = batch['label'], batch['input_ids'], batch['attention_mask']
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            predictions.append(logits.softmax(dim=1).cpu().numpy())
    
    y_score = predictions[0][:,1]


    # Calculate TPR and FPR
    fpr_semi, tpr_semi, thresholds_semi = roc_curve(y_true, y_score)

    # Calculate AUC
    roc_auc_semi = auc(fpr_semi, tpr_semi)

    PlotHistogram(y_true, y_score, hist_name_2)
    # Plot ROC curve
    plt.rcParams["figure.figsize"] = (10, 10)

    plt.plot(fpr_init, tpr_init, lw=1, alpha=0.8, label='Transformer (AUC = %0.2f)' % roc_auc_init)
    plt.plot(fpr_semi, tpr_semi, lw=1, alpha=0.8, label='SS Transformer (AUC = %0.2f)' % roc_auc_semi)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_plot_name)
    plt.close()