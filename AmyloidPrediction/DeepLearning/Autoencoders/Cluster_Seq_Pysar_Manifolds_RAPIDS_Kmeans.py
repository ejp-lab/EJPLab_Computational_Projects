from cuml import UMAP
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cuml.metrics.cluster import silhouette_score
from tqdm import tqdm
from cuml import DBSCAN
import optuna
from cuml import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import colors

import seaborn as sns

# Cluster_Seq_pysar_Manifold_RAPIDS_Kmeans.py - Combines the sequence and pySAR manifolds and performs kmeans clustering. Generates manifold figure. Splits datasets into testing and training. (RAPIDS)  
# To run: python Cluster_Seq_Pysar_Manfiolds_Rapids_Kmeans.py  

sns.set_palette('colorblind')
plt.rc('legend', fontsize=13)
plt.rcParams["figure.figsize"] = (10,10)
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
SEED = 42
np.random.seed(SEED)

if __name__ == "__main__":

    DIVISOR = 10

    pysar_folder = "Pysar_Embeddings"
    seq_folder = "Seq_Embeddings"

    with open(f"{pysar_folder}/Training_Set_Embeddings_Pysar.pickle", 'rb') as f:
        train_pysar = pickle.load(f)

    with open(f"{pysar_folder}/Validation_Set_Embeddings_Pysar.pickle", 'rb') as f:
        val_pysar = pickle.load(f)

    with open(f"{pysar_folder}/Testing_Set_Embeddings_Pysar.pickle", 'rb') as f:
        test_pysar = pickle.load(f)

    with open(f"{seq_folder}/Training_Set_Embeddings_Seq.pickle", 'rb') as f:
        train_seq = pickle.load(f)

    with open(f"{seq_folder}/Validation_Set_Embeddings_Seq.pickle", 'rb') as f:
        val_seq = pickle.load(f)

    with open(f"{seq_folder}/Testing_Set_Embeddings_Seq.pickle", 'rb') as f:
        test_seq = pickle.load(f)
    
    train_pysar = np.concatenate([train_pysar, train_seq], axis=1)
    val_pysar= np.concatenate([val_pysar, val_seq], axis=1)
    test_pysar = np.concatenate([test_pysar, test_seq], axis=1)

    train = train_pysar[np.random.choice(train_pysar.shape[0], int(train_pysar.shape[0]/DIVISOR), replace=False), :]

    concat_data = np.concatenate([val_pysar, test_pysar], axis=0)

    scores = {}
    for i in tqdm(range(2,100, 2)):
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(train_seq)
        labels = kmeans.labels_
        score = davies_bouldin_score(train_seq, labels)
        scores.update({i:score})

    plt.plot(scores.keys(), scores.values())
    plt.show()

    n_clusters = input("Chosen number of clusters: ")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(train_pysar)

    umap_obj = UMAP(n_neighbors=100)
    umap_embeddings_obj = umap_obj.fit(train)
    umap_test_emb = umap_obj.transform(test_pysar)
    umap_val_emb = umap_obj.transform(val_pysar)

    umap_train_emb = umap_embeddings_obj.embedding_

    def remove_outliers(arr, labels, k):
        mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0)
        mask = np.all(np.abs((arr - mu) / sigma) < k, axis=1)
        labels = labels[mask]
        arr = arr[mask]
        return arr, labels

    test_labels = pd.read_csv("Test_Set.csv")['Class'].to_numpy()
    val_labels = pd.read_csv("Validation_Set.csv")['Class'].to_numpy()
    train_labels = kmeans.predict(train)

    # Removes error due to gpu
    umap_train_emb, train_labels = remove_outliers(umap_train_emb, train_labels, 3)
    umap_test_emb, test_labels = remove_outliers(umap_test_emb, test_labels, 3)
    umap_val_emb, val_labels = remove_outliers(umap_val_emb, val_labels, 3)
    # for the new train set
    

    num_labels = len(set(train_labels))

    cmap = plt.get_cmap('gray', num_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_labels):
        ax.scatter(umap_train_emb[train_labels == i, 0], umap_train_emb[train_labels == i, 1], s=0.001, color=cmap(i - 1), alpha=0.5)

    c = ['r' if i == 0 else 'b' for i in test_labels]
    ax.scatter(umap_test_emb[:, 0], umap_test_emb[:, 1], s=5, color=c)

    c = ['r' if i == 0 else 'b' for i in val_labels]
    ax.scatter(umap_val_emb[:, 0], umap_val_emb[:, 1], s=5, color=c)

    pickle.dump(umap_train_emb, open("Joint_Umap_Train_Embeddings.pickle", 'wb'))
    pickle.dump(umap_val_emb, open("Joint_Umap_Val_Embeddings.pickle", 'wb'))
    pickle.dump(umap_test_emb, open("Joint_Umap_Test_Embeddings.pickle", 'wb'))

    #plt.legend(['Synthetic Datapoint', 'Test Positive', 'Test Negative', 'Train Positive', 'Train Negative'])
    
    fig.tight_layout()
    plt.savefig("Combined_Embedding_Space.png")
    plt.close()

    test_dataset = pd.read_csv("Test_Set.csv")
    val_dataset = pd.read_csv("Validation_Set.csv")
    total_dataset = pd.concat([test_dataset, val_dataset], axis=0)

    X = total_dataset['Sequence'].to_numpy()
    y = kmeans.predict(concat_data)
    labels = total_dataset['Class'].to_numpy()

    sss = StratifiedShuffleSplit(train_size=1122)
    train_index, test_index = next(sss.split(X, y))

    train_X = X[train_index]
    test_X = X[test_index]

    train_y = labels[train_index]
    test_y = labels[test_index]

    train_set = pd.DataFrame(np.array([train_X, train_y]).T, columns=['Sequence', 'label'])
    test_set = pd.DataFrame(np.array([test_X, test_y]).T, columns=['Sequence', 'label'])

    train_set.to_csv("Clustered_Train_Set.csv", index=None)
    test_set.to_csv("Clustered_Test_Set.csv", index=None)


