import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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

    # Simple frequency of classes plot
    train_dataset = pd.read_csv("Clustered_Train_Set.csv")
    test_dataset = pd.read_csv("Clustered_Test_Set.csv")
    total_set = pd.concat([train_dataset, test_dataset], axis=0)

    positive_df = total_set[total_set['label'] == 1]
    negative_df = total_set[total_set['label'] != 1]

    positives = len(positive_df)
    negatives = len(negative_df)

    plt.bar(['Negative', 'Positive'], [negatives, positives])
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("Class_Distribution_Bar_Chart.png")
    plt.close()

    # Amino acid positive vs negative class abundance
    amino_acids_pos = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}
    amino_acids_neg = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

    for sequence in positive_df.Sequence:
        for letter in sequence:
            amino_acids_pos[letter] += 1
    
    for sequence in negative_df.Sequence:
        for letter in sequence:
            amino_acids_neg[letter] += 1

    total_pos = sum(amino_acids_pos.values())
    total_neg = sum(amino_acids_neg.values())

    # Normalization
    amino_acids_pos = {key: value/total_pos for key, value in amino_acids_pos.items()}
    amino_acids_neg = {key: value/total_neg for key, value in amino_acids_neg.items()}

    # Sorting to highest frequency positive
    amino_acids_pos = dict(sorted(amino_acids_pos.items(), key=lambda x: x[1], reverse=True))
    amino_acids_neg = dict(sorted(amino_acids_neg.items(), key=lambda x: amino_acids_pos.get(x[0]), reverse=True))
    ticks = np.arange(len(amino_acids_pos.keys()))
    bar_width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar(ticks - bar_width, amino_acids_pos.values(), width=bar_width, label="Positive Class", color="b", alpha=0.75)
    ax.bar(ticks, amino_acids_neg.values(), width=bar_width, label="Negative Class", color="r", alpha=0.75)
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Normalized Frequency')
    ax.set_xticks(ticks - bar_width/2)
    ax.set_xticklabels(amino_acids_pos.keys())
    legend = ax.legend(loc='upper right')
    legend.get_frame().set_linewidth(0.0)
    fig.tight_layout()

    plt.savefig("Amino_Acid_Composition.png")
    plt.close()
    
    # Same as above but in groupings
    def get_amino_acid_groupings(sequences):
        amino_acid_grouping = {
            "Nonpolar": ["A", "I", "L", "M", "V"],
            "Polar": ["C", "N", "Q", "S", "T"],
            "Anionic": ["D", "E"],
            "Cationic": ["H", "K", "R"],
            "Glycine": ["G"],
            "Proline": ["P"],
            "Aromatic": ["W", "Y", "F"]
        }

        # create an empty dictionary to hold the counts
        counts = {
            "Nonpolar": 0,
            "Polar": 0,
            "Anionic": 0,
            "Cationic": 0,
            "Glycine": 0,
            "Proline": 0,
            "Aromatic": 0
        }

        # loop through the sequence and count the amino acids
        for sequence in sequences:
            for residue in sequence:
                for key, value in amino_acid_grouping.items():
                    if residue in value:
                        counts[key] += 1
        
        total = sum(counts.values())
        counts = {key: value/total for key, value in counts.items()}
        
        return counts
    
    positive_sequences = positive_df.Sequence.to_list()
    negative_sequences = negative_df.Sequence.to_list()

    grouped_positive = get_amino_acid_groupings(positive_sequences)
    grouped_negative = get_amino_acid_groupings(negative_sequences)

    grouped_positive = dict(sorted(grouped_positive.items(), key=lambda x: x[1], reverse=True))
    grouped_negative = dict(sorted(grouped_negative.items(), key=lambda x: grouped_positive.get(x[0]), reverse=True))

    ticks = np.arange(len(grouped_positive.keys()))
    bar_width = 0.35

    fig, ax = plt.subplots()
    ax.bar(ticks - bar_width, grouped_positive.values(), width=bar_width, label="Positive Class", color="b", alpha=0.75)
    ax.bar(ticks, grouped_negative.values(), width=bar_width, label="Negative Class", color="r", alpha=0.75)
    ax.set_xlabel('Amino Acid Type')
    ax.set_ylabel('Normalized Frequency')
    ax.set_xticks(ticks - bar_width/2)
    ax.set_xticklabels(grouped_positive.keys())
    legend = ax.legend(loc='upper right')
    legend.get_frame().set_linewidth(0.0)
    fig.tight_layout()

    plt.savefig("Amino_Acid_Composition_Grouped.png")
    plt.close()

    # Combine previous 2 plots

    fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [2.4, 1]})

    ticks = np.arange(len(amino_acids_pos.keys()))
    ax1 = axes[0]
    ax1.bar(ticks - bar_width, amino_acids_pos.values(), width=bar_width, label="Positive Class", color="b", alpha=0.75)
    ax1.bar(ticks, amino_acids_neg.values(), width=bar_width, label="Negative Class", color="r", alpha=0.75)
    ax1.set_xlabel('Amino Acid')
    ax1.set_ylabel('Normalized Frequency')
    ax1.set_xticks(ticks - bar_width/2)
    ax1.set_xticklabels(amino_acids_pos.keys())
    #legend1 = ax1.legend(loc='upper right')
    #legend1.get_frame().set_linewidth(0.0)

    ticks = np.arange(len(grouped_positive.keys()))

    ax2 = axes[1]
    ax2.bar(ticks - bar_width, grouped_positive.values(), width=bar_width, label="Positive Class", color="b", alpha=0.75)
    ax2.bar(ticks, grouped_negative.values(), width=bar_width, label="Negative Class", color="r", alpha=0.75)
    ax2.set_xlabel('Amino Acid Type')
    #ax2.set_ylabel('Normalized Frequency')
    ax2.set_xticks(ticks - bar_width/2)
    ax2.set_xticklabels(grouped_positive.keys(), rotation=45, ha='center')
    legend2 = ax2.legend(loc='upper right')
    legend2.get_frame().set_linewidth(0.0)

    fig.tight_layout()
    plt.savefig("Amino_Acid_Composition_combined.png")
    plt.close()