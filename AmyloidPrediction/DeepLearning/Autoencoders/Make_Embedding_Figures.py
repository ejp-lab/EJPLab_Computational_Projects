import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from cuml import UMAP
import pandas as pd
import pickle
import numpy as np
from cuml import KMeans
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm

# Make_Embedding_Figures.py - Makes UMAP embeddings for sequence and pySAR manifolds separately. (RAPIDS) 
# To Run: python Make_Embedding_Figures.py  

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

    train_pysar = train_pysar[np.random.choice(train_pysar.shape[0], int(train_pysar.shape[0]/DIVISOR), replace=False), :]
    
    scores = {}
    for i in tqdm(range(2,100, 2)):
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(train_pysar)
        labels = kmeans.labels_
        score = davies_bouldin_score(train_pysar, labels)
        scores.update({i:score})

    plt.plot(scores.keys(), scores.values())
    plt.show()
    
    n_clusters = int(input("Choose N Clusters: "))

    # pysar Sets
    umap_obj = UMAP(n_neighbors=15)
    umap_embeddings_obj = umap_obj.fit(train_pysar)

    umap_train_emb = umap_embeddings_obj.embedding_
    umap_test_emb = umap_obj.transform(test_pysar)
    umap_val_emb = umap_obj.transform(val_pysar)

    test_labels = pd.read_csv("Test_Set.csv")['Class'].to_numpy()
    val_labels = pd.read_csv("Validation_Set.csv")['Class'].to_numpy()

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(train_pysar)
    train_labels = kmeans.predict(train_pysar)

    def remove_outliers(arr, labels, k):
        mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0)
        mask = np.all(np.abs((arr - mu) / sigma) < k, axis=1)
        labels = labels[mask]
        arr = arr[mask]
        return arr, labels

    umap_train_emb, train_labels = remove_outliers(umap_train_emb, train_labels, 3)
    umap_test_emb, test_labels = remove_outliers(umap_test_emb, test_labels, 3)
    umap_val_emb, val_labels = remove_outliers(umap_val_emb, val_labels, 3)

    num_labels = len(set(train_labels))

    cmap = plt.get_cmap('gray', num_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_labels):
        ax.scatter(umap_train_emb[train_labels == i, 0], umap_train_emb[train_labels == i, 1], s=0.001, color=cmap(i - 1), alpha=0.5)
    
    c = ['r' if i == 0 else 'b' for i in test_labels]

    ax.scatter(umap_test_emb[:, 0], umap_test_emb[:, 1], s=5, color=c, alpha=0.5)

    c = ['r' if i == 0 else 'b' for i in val_labels]

    ax.scatter(umap_val_emb[:, 0], umap_val_emb[:, 1], s=5, color=c, alpha=0.5)

    pickle.dump(umap_train_emb, open("UMAP_Pysar_Embeddings_Train.pickle", 'wb'))
    pickle.dump(umap_val_emb, open("UMAP_Pysar_Embeddings_Val.pickle", 'wb'))
    pickle.dump(umap_test_emb, open("UMAP_Pysar_Embeddings_Test.pickle", 'wb'))

    fig.tight_layout()
    plt.savefig("Pysar_Embedding_Space.png")
    plt.close()

    train_seq = train_seq[np.random.choice(train_seq.shape[0], int(train_seq.shape[0]/DIVISOR), replace=False), :]

    # Sequence Sets
    scores = {}
    for i in tqdm(range(2,100, 2)):
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(train_seq)
        labels = kmeans.labels_
        score = davies_bouldin_score(train_seq, labels)
        scores.update({i:score})

    plt.plot(scores.keys(), scores.values())
    plt.show()
    
    n_clusters = int(input("Choose N Clusters: "))

    # pysar Sets
    umap_obj = UMAP(n_neighbors=100)
    umap_embeddings_obj = umap_obj.fit(train_seq)

    umap_train_emb = umap_embeddings_obj.embedding_
    umap_test_emb = umap_obj.transform(test_seq)
    umap_val_emb = umap_obj.transform(val_seq)

    test_labels = pd.read_csv("Test_Set.csv")['Class'].to_numpy()
    val_labels = pd.read_csv("Validation_Set.csv")['Class'].to_numpy()

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(train_seq)
    train_labels = kmeans.predict(train_seq)

    def remove_outliers(arr, labels, k):
        mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0)
        mask = np.all(np.abs((arr - mu) / sigma) < k, axis=1)
        labels = labels[mask]
        arr = arr[mask]
        return arr, labels
    
    umap_train_emb, train_labels = remove_outliers(umap_train_emb, train_labels, 3)
    umap_test_emb, test_labels = remove_outliers(umap_test_emb, test_labels, 3)
    umap_val_emb, val_labels = remove_outliers(umap_val_emb, val_labels, 3)

    num_labels = len(set(train_labels))

    cmap = plt.get_cmap('gray', num_labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(num_labels):
        ax.scatter(umap_train_emb[train_labels == i, 0], umap_train_emb[train_labels == i, 1], s=0.001, color=cmap(i - 1), alpha=0.5)

    c = ['r' if i == 0 else 'b' for i in test_labels]
    ax.scatter(umap_test_emb[:, 0], umap_test_emb[:, 1], s=5, color=c, alpha=0.5)

    c = ['r' if i == 0 else 'b' for i in val_labels]
    ax.scatter(umap_val_emb[:, 0], umap_val_emb[:, 1], s=5, color=c, alpha=0.5)

    pickle.dump(umap_train_emb, open("UMAP_Seq_Embeddings_Train.pickle", 'wb'))
    pickle.dump(umap_val_emb, open("UMAP_Seq_Embeddings_Val.pickle", 'wb'))
    pickle.dump(umap_test_emb, open("UMAP_Seq_Embeddings_Test.pickle", 'wb'))

    fig.tight_layout()
    plt.savefig("Seq_Embedding_Space.png")
    plt.close()