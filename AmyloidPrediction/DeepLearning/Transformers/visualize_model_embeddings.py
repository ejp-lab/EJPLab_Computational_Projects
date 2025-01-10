from cuml import PCA
import pickle
import numpy as np
from cuml import UMAP
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import seaborn as sns

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

if __name__ == "__main__":

    big_embeddings = pickle.load(open("Embedding_of_Big_Dataset.pickle",'rb'))
    train_embeddings = pickle.load(open("Embedding_of_Transformer_Train.pickle",'rb'))
    test_embeddings = pickle.load(open("Embedding_of_Transformer_Test.pickle", 'rb'))

    combined_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    
    umap = UMAP(n_neighbors=100)
    umap.fit(big_embeddings)
    umap_embeddings = umap.transform(combined_embeddings)

    train = pd.read_csv("TrainingDataset.csv")
    test = pd.read_csv("TestingDataset.csv")
    total_df = pd.concat([train, test], axis=0)
    labels = train['label'].to_list() + test['label'].to_list()

    c = ['r' if i == 0 else 'b' for i in labels]

    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5, c=c)
    plt.show()
