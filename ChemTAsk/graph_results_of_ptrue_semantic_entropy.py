import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data for QADebertaEntailment
with open('results_QADebertaEntailment.pickle', 'rb') as f:
    results_QADeberta = pickle.load(f)
    results_QADeberta_results = results_QADeberta[0]
    results_QADeberta_all_uncertainties = results_QADeberta[1]
    results_QADeberta_question_uncertainties = results_QADeberta[2]

# Load data for PTrueOriginalBaseline
with open('results_PTrueOriginalBaseline.pickle', 'rb') as f:
    results_PTrueBaseline = pickle.load(f)
    results_PTrueBaseline_results = results_PTrueBaseline[0]
    results_PTrueBaseline_all_uncertainties = results_PTrueBaseline[1]
    results_PTrueBaseline_question_uncertainties = results_PTrueBaseline[2]

# Combine all uncertainties to find global min and max
all_uncertainties_combined = (
    results_PTrueBaseline_all_uncertainties +
    results_QADeberta_all_uncertainties +
    results_PTrueBaseline_question_uncertainties +
    results_QADeberta_question_uncertainties
)

# Find global x-axis range
x_min = min(all_uncertainties_combined)
x_max = max(all_uncertainties_combined)

# Define common bins
bins = np.linspace(x_min, x_max, num=21)  # 20 bins

# Set Seaborn style and context
sns.set(style='whitegrid', context='talk', font_scale=1.2)

# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Top Left: PTrueOriginalBaseline with all_uncertainties
sns.histplot(
    results_PTrueBaseline_all_uncertainties,
    bins=bins,
    ax=axs[0, 0],
    kde=False,
    color='skyblue',
    edgecolor='black'
)
axs[0, 0].set_title('P(True)\nAll Uncertainties', fontsize=16)
axs[0, 0].set_xlabel('')
axs[0, 0].set_ylabel('Frequency', fontsize=14)
axs[0, 0].set_xlim([x_min, x_max])

# Top Right: QADebertaEntailment with all_uncertainties
sns.histplot(
    results_QADeberta_all_uncertainties,
    bins=bins,
    ax=axs[0, 1],
    kde=False,
    color='lightgreen',
    edgecolor='black'
)
axs[0, 1].set_title('Semantic Entropy\nAll Uncertainties', fontsize=16)
axs[0, 1].set_xlabel('')
axs[0, 1].set_ylabel('')
axs[0, 1].set_xlim([x_min, x_max])
axs[0, 1].set_yticklabels([])

# Bottom Left: PTrueOriginalBaseline with question uncertainties
sns.histplot(
    results_PTrueBaseline_question_uncertainties,
    bins=bins,
    ax=axs[1, 0],
    kde=False,
    color='skyblue',
    edgecolor='black'
)
axs[1, 0].set_title('P(True)\nQuestion Uncertainties', fontsize=16)
axs[1, 0].set_xlabel('Uncertainty', fontsize=14)
axs[1, 0].set_ylabel('Frequency', fontsize=14)
axs[1, 0].set_xlim([x_min, x_max])

# Bottom Right: QADebertaEntailment with question uncertainties
sns.histplot(
    results_QADeberta_question_uncertainties,
    bins=bins,
    ax=axs[1, 1],
    kde=False,
    color='lightgreen',
    edgecolor='black'
)
axs[1, 1].set_title('Semantic Entropy\nQuestion Uncertainties', fontsize=16)
axs[1, 1].set_xlabel('Uncertainty', fontsize=14)
axs[1, 1].set_ylabel('')
axs[1, 1].set_xlim([x_min, x_max])
axs[1, 1].set_yticklabels([])

# Adjust tick parameters for all axes
for ax_row in axs:
    for ax in ax_row:
        ax.tick_params(axis='both', which='major', labelsize=12)

# Remove internal axis labels
axs[0, 0].set_xlabel('')
axs[0, 1].set_xlabel('')
axs[0, 1].set_ylabel('')
axs[1, 1].set_ylabel('')

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig('uncertainties_histograms.png', dpi=300)
plt.show()
