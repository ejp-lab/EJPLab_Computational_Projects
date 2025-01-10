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
from collections import Counter
import seaborn as sns
import sys


# 
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

    n_clusters = int(sys.argv[1])

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(train_pysar)

    test_labels = pd.read_csv("Test_Set.csv")['Class'].to_numpy()
    val_labels = pd.read_csv("Validation_Set.csv")['Class'].to_numpy()
    train_labels = kmeans.predict(train)

    test_dataset = pd.read_csv("Test_Set.csv")
    val_dataset = pd.read_csv("Validation_Set.csv")
    total_dataset = pd.concat([test_dataset, val_dataset], axis=0)

    X = total_dataset['Sequence'].to_numpy()
    y = kmeans.predict(concat_data)
    labels = total_dataset['Class'].to_numpy()

    def find_common_motifs_normalized(sequences, k=1):
        all_kmers = []
        for seq in sequences:
            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
            all_kmers.extend(kmers)
        kmer_counts = Counter(all_kmers)
        total_sequences = len(sequences)
        most_common_kmers = [(kmer, count / total_sequences) for kmer, count in kmer_counts.most_common(5)]
        most_common_kmers_sorted = sorted(most_common_kmers, key=lambda x: x[1], reverse=True)
        return most_common_kmers_sorted

    motifs_by_cluster_normalized = {}
    for cluster in range(n_clusters):
        cluster_sequences = X[y == cluster]
        common_motifs_normalized = find_common_motifs_normalized(cluster_sequences, int(sys.argv[2]))
        motifs_by_cluster_normalized[cluster] = common_motifs_normalized
        print(f"Cluster {cluster}: {common_motifs_normalized}")

def create_motif_dataframe_with_class_labels_and_proportions(motifs_by_cluster_normalized, X, y, labels):
    data = []
    total_sequences = len(X)
    
    for cluster in range(len(motifs_by_cluster_normalized)):
        cluster_sequences = X[y == cluster]
        cluster_labels = labels[y == cluster]
        cluster_size = len(cluster_sequences)
        
        positive_prop = np.mean(cluster_labels == 1)
        negative_prop = np.mean(cluster_labels == 0)
        cluster_prop = cluster_size / total_sequences
        
        row = {'Cluster': cluster, 'Proportion': cluster_prop, 'Positive': positive_prop, 'Negative': negative_prop}
        for kmer, freq in motifs_by_cluster_normalized[cluster]:
            row[kmer] = freq
        data.append(row)
        
    df = pd.DataFrame(data)
    return df

def plot_common_motifs_heatmap_with_class_labels_and_proportions(motifs_by_cluster_normalized, X, y, labels):
    df = create_motif_dataframe_with_class_labels_and_proportions(motifs_by_cluster_normalized, X, y, labels)
    df = df.set_index('Cluster')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Frequency'}, annot_kws={'size': 10})
    plt.title('Normalized Common Motifs, Class Proportions, and Cluster Proportions')
    plt.xlabel('Motifs, Class Proportions, and Cluster Proportions')
    plt.ylabel('Cluster')
    plt.show()

plot_common_motifs_heatmap_with_class_labels_and_proportions(motifs_by_cluster_normalized, X, y, labels)


