from p_true import *
from huggingface_models import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from sklearn.metrics import roc_curve, auc
import sys

sns.set(style="whitegrid", context="talk")

# Define the models and datasets
models = ['gpt-4o', 'gpt-4o-mini', 'llama-8B', 'llama-70B']
datasets = ['train', 'test']

# Create individual histogram plots for each model and dataset
for dataset in datasets:
    for model_name in models:
        # Load the data
        try:
            with open(f'{dataset}_{model_name}_stats.pickle', 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"File {dataset}_{model_name}_stats.pickle not found.")
            continue

        probabilities = data['probabilities']
        answers = data['answers']
        correct_answers = data['correct_answers']

        # Compute grades
        grades = [1 if a in correct_answers[idx] else 0 for idx, a in enumerate(answers)]

        # Separate probabilities into correct and incorrect
        probs_correct = [prob for prob, grade in zip(probabilities, grades) if grade == 1]
        probs_incorrect = [prob for prob, grade in zip(probabilities, grades) if grade == 0]

        # Define the bins for the histogram
        bins = np.linspace(min(probabilities), max(probabilities), 30)

        # Create a figure and axis object
        plt.figure(figsize=(14, 8))

        # Histogram for correct answers
        sns.histplot(
            probs_correct,
            bins=bins,
            color='green',
            label='Correct',
            alpha=0.5,
            edgecolor='black'
        )

        # Histogram for incorrect answers
        sns.histplot(
            probs_incorrect,
            bins=bins,
            color='blue',
            label='Incorrect',
            alpha=0.5,
            edgecolor='black'
        )

        # Customize the plot
        plt.title(f'Distribution of Model Probabilities by Answer Correctness\nModel: {model_name}, Dataset: {dataset.capitalize()}', fontsize=20, fontweight='bold')
        plt.xlabel('Probability', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)

        # Set x and y ticks font size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Add legend with custom settings
        plt.legend(title='Answer Correctness', title_fontsize=16, fontsize=14)

        # Add gridlines
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{dataset}_probability_distribution_{model_name}.png')
        plt.close()

# Create a big plot for the AUROC curves with two subplots
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

for idx, dataset in enumerate(datasets):
    ax = axs[idx]
    ax.set_title(f'{dataset.capitalize()} Dataset', fontsize=20)

    for model_name in models:
        # Load the data
        try:
            with open(f'{dataset}_{model_name}_stats.pickle', 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"File {dataset}_{model_name}_stats.pickle not found.")
            continue

        probabilities = data['probabilities']
        answers = data['answers']
        correct_answers = data['correct_answers']

        # Compute grades
        grades = [1 if a in correct_answers[idx] else 0 for idx, a in enumerate(answers)]

        # Compute False Positive Rate and True Positive Rate
        fpr, tpr, thresholds = roc_curve(grades, probabilities)

        # Compute the area under the ROC curve
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        ax.plot(fpr, tpr, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot the diagonal line (no-skill classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Customize the plot
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=16)
    if idx == 0:
        ax.set_ylabel('True Positive Rate', fontsize=16)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    # Set x and y ticks font size
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add legend with custom settings
    ax.legend(loc='lower right', fontsize=16)

    # Add gridlines
    ax.grid(visible=True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

# Set the main title

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('AUROC_all_models.png')
plt.close()
