import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from krippendorff import alpha
from scipy.stats import wilcoxon

# Load the CSV file
df = pd.read_csv('deidentified_dataset_final.csv')

def create_word_cloud_with_top_words(text_series, title, colors):
    # Combine all texts into one large string
    combined_text = " ".join(review for review in text_series.dropna())
    
    # Define a custom color map from the provided list of colors
    custom_color_map = LinearSegmentedColormap.from_list("custom_spectrum", colors)
    
    # Create and generate a word cloud image using the custom color map
    wordcloud = WordCloud(background_color='white', max_words=200, contour_width=3,
                          colormap=custom_color_map).generate(combined_text)
    
    # Get the frequencies of the words
    word_counts = wordcloud.process_text(text=combined_text)  # Actual counts
    
    # Get top 20 words by count
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    words, counts = zip(*top_words)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8)) 
    
    # Left subplot: Word Cloud
    axes[0].imshow(wordcloud, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title(title, fontsize=20)
    
    # Get colors for bars from the colormap to create a gradient
    colors_for_bars = [custom_color_map(i / len(words)) for i in range(len(words))]
    
    # Adjust bar thickness by setting the height parameter
    bar_height = 0.8  # Adjust as needed to make bars thinner
    
    y_positions = np.arange(len(words))
    axes[1].barh(y_positions, counts, color=colors_for_bars, height=bar_height)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(words,fontsize=14)
    axes[1].invert_yaxis()  # Largest counts at the top
    axes[1].set_xlabel('Counts',fontsize=14)
    axes[1].set_title('Top 20 Words',fontsize=20)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(title + "_with_top_words.png")
    plt.show()
    plt.close()

# Preprocessing
df['Body'] = df['Body'].str.replace('<FILTERED>', '', regex=False)  # Removing Filtered
df = df[df['Sender'] != 'Person_325c7c08']  # Remove specific sender because of formatting issues

# Define color spectrums for each word cloud
cool_colors = ['teal', 'deepskyblue', 'blue', 'purple']
warm_colors = ['pink', 'red', 'darkred', 'maroon']

# Generate word clouds with the top 20 words and gradient bar charts
create_word_cloud_with_top_words(df['Response'], 'Word Cloud for Response Column', cool_colors)
create_word_cloud_with_top_words(df['Body'], 'Word Cloud for Body Column', warm_colors)

def plot_sender_pie_chart(df):
    # Count the occurrences of each sender
    sender_counts = df['Sender'].value_counts()
    
    # If there are more than 10 unique senders, group the rest as 'Other'
    if len(sender_counts) > 5:
        top_senders = sender_counts.iloc[:5]
        others_sum = sender_counts.iloc[5:].sum()
        # Append the sum of all others as a new row labeled 'Other'
        top_senders['Other'] = others_sum
    else:
        top_senders = sender_counts
    
    # Plot a pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(top_senders, labels=top_senders.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Top 5 Senders and Others')
    plt.show()

plot_sender_pie_chart(df)

def plot_weekday_distribution(df, timestamp_column):
    """
    Plot a pie chart showing the distribution of messages by day of the week.
    :param df: DataFrame containing the timestamps.
    :param timestamp_column: Name of the column containing timestamp data.
    """
    # Convert the timestamp column to datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Extract the day of the week (Monday=0, Sunday=6)
    df['Day of Week'] = df[timestamp_column].dt.dayofweek
    
    # Map day of the week from number to name
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['Day of Week Name'] = df['Day of Week'].map(days)
    
    # Count occurrences of each day
    weekday_counts = df['Day of Week Name'].value_counts()
    
    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Messages by Day of Week')
    plt.axis('equal')  # Ensure the pie chart is circular
    plt.show()

plot_weekday_distribution(df, 'Timestamp')

usage_counts = df['Sender'].value_counts().reset_index()
usage_counts.columns = ['Codename', 'Usage_Frequency']

def plot_weekly_sender_usage_line(df, timestamp_column, top_n_senders=None):
    """
    Plot the weekly usage of each sender over time as a line graph.
    :param df: DataFrame containing the data.
    :param timestamp_column: Name of the column containing timestamp data.
    :param top_n_senders: (Optional) Number of top senders to include. Others will be grouped as 'Other'.
    """
    
    # Ensure 'Timestamp' is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Create a 'Week' column representing the week starting date
    df['Week'] = df[timestamp_column].dt.to_period('W').apply(lambda r: r.start_time)

    # Count the number of messages per sender per week
    weekly_usage = df.groupby(['Week', 'Sender']).size().reset_index(name='Message_Count')

    # If top_n_senders is specified, group other senders into 'Other'
    if top_n_senders is not None:
        # Get the top N senders
        top_senders = df['Sender'].value_counts().nlargest(top_n_senders).index
        # Replace sender codenames not in top_senders with 'Other'
        weekly_usage['Sender'] = weekly_usage['Sender'].apply(
            lambda x: x if x in top_senders else 'Other'
        )
        # Recalculate the weekly usage after grouping 'Other' senders
        weekly_usage = weekly_usage.groupby(['Week', 'Sender']).sum().reset_index()

    # Pivot the table to have 'Week' as index and 'Sender_Codename' as columns
    usage_pivot = weekly_usage.pivot(index='Week', columns='Sender', values='Message_Count').fillna(0)

    # Sort the weeks in ascending order
    usage_pivot = usage_pivot.sort_index()

    # Plot the data as a line graph
    ax = usage_pivot.plot(kind='line', figsize=(15, 7), marker='o', linewidth=2)

    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Number of Messages', fontsize=14)
    plt.title('Weekly Message Usage by Sender', fontsize=16)
    plt.legend(title='Sender Codename', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Senders_Usage_Over_Time.png')
    plt.show()
    plt.close()

plot_weekly_sender_usage_line(df, 'Timestamp', 5)

### Krippendorfs alpha ###
sns.set(style="whitegrid")
palette = sns.color_palette("coolwarm", 8)

file_path = "deidentified_reviewer_scores.csv"
data = pd.read_csv(file_path)
# Extract ratings columns for each question
questions = [f"Q{i}" for i in range(1, 6)]

reviewer_columns = [col for col in data.columns if 'Reviewer' in col]

# Getting unique reviewer names from all these columns
unique_reviewers = pd.unique(data[reviewer_columns].values.ravel('K'))

# Removing any NaN values if present
unique_reviewers = [reviewer for reviewer in unique_reviewers if pd.notna(reviewer)]
# code names
reviewer_nums = {reviewer:i for i,reviewer in enumerate(unique_reviewers)}
num_reviewers = len(unique_reviewers)
# Initialize a dictionary to hold ratings for each question
ratings = np.full((len(unique_reviewers),len(data), 5, 2), np.nan) 
# Shape is for reviewers, number of data points, five questions, 2 types of responders

# Populate the dictionary with the ratings
for qidx, q in enumerate(questions):
    chatgpt_cols = [col for col in data.columns if col.startswith(f"{q}_ChatGPT_Replicate") and not col.endswith("Reviewer")]
    ta_cols = [col for col in data.columns if col.startswith(f"{q}_TA_Replicate") and not col.endswith("Reviewer")]
    
    chatgpt_ratings = data[chatgpt_cols].values
    ta_ratings = data[ta_cols].values

    graders_cols_chatgpt = [col for col in data.columns if col.startswith(f"{q}_ChatGPT_Replicate") and col.endswith("Reviewer")]
    graders_cols_ta = [col for col in data.columns if col.startswith(f"{q}_TA_Replicate") and col.endswith("Reviewer")]

    graders_chatgpt = data[graders_cols_chatgpt].to_numpy()
    graders_ta = data[graders_cols_ta].to_numpy()

    for i, row in enumerate(chatgpt_ratings):
        for j, rate in enumerate(row):
            rater_idx = reviewer_nums[graders_chatgpt[i][j]]
            ratings[rater_idx][i][qidx][0] = rate

    for i, row in enumerate(ta_ratings):
        for j, rate in enumerate(row):
            rater_idx = reviewer_nums[graders_ta[i][j]]
            ratings[rater_idx][i][qidx][1] = rate

alphas = np.zeros((len(questions), 2))
for i in range(alphas.shape[0]):
    for j in range(alphas.shape[1]):
        alphas[i][j] = alpha(reliability_data=ratings[:,:,i,j], level_of_measurement='interval')

rows = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
columns = ['ChemTAsk', 'TA']

# Creating the bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(rows))

for i, column in enumerate(columns):
    if i % 2 == 0:
        color = -1
    else:
        color = 0
    ax.bar(index + i * bar_width, alphas[:, i], bar_width, label=column, color=palette[color])

# Adding labels and title
ax.set_xlabel('Questions')
ax.set_ylabel('Alpha Values')
ax.set_title('Alpha Values for ChemTAsk and TA')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(rows)

plt.savefig("k_alphas_chatgpt_TA.png")
plt.close()

df = pd.read_csv("deidentified_reviewer_scores.csv")

for i in range(1, 6):  # Q1 to Q5
    for source in ['ChatGPT', 'TA']:
        replicate_columns = [f'Q{i}_{source}_Replicate_{j}' for j in range(1, 4)]
        
        df[f'Q{i}_{source}_Mean'] = df[replicate_columns].mean(axis=1)
        df[f'Q{i}_{source}_std'] = df[replicate_columns].std(axis=1)

indices = df.index
chatgpt_means = df[[f'Q{i}_ChatGPT_Mean' for i in range(1, 6)]].mean(axis=1)
reviewer_means = df[[f'Q{i}_TA_Mean' for i in range(1, 6)]].mean(axis=1)

chatgpt_std = df[[f'Q{i}_ChatGPT_std' for i in range(1, 6)]].mean(axis=1)
reviewer_std = df[[f'Q{i}_TA_std' for i in range(1, 6)]].mean(axis=1)

# Calculate global means and standard deviations for ChatGPT and Reviewer
global_chatgpt_mean = chatgpt_means.mean()
global_reviewer_mean = reviewer_means.mean()

global_chatgpt_std = chatgpt_std.mean()
global_reviewer_std = reviewer_std.mean()

### Figure 2 ####
def plot_difference(df):

    indices = df.index
    differences = df[[f'Q{i}_TA_Mean' for i in range(1, 6)]].mean(axis=1) - df[[f'Q{i}_ChatGPT_Mean' for i in range(1, 6)]].mean(axis=1)
    # Sort by differences
    sorted_indices = [x for _, x in sorted(zip(differences, indices), reverse=True)]
    sorted_differences = sorted(differences, reverse=True)

    # Set Seaborn style and color palette
    sns.set(style="whitegrid")
    palette = sns.color_palette("coolwarm", len(differences))

    fig, ax = plt.subplots(figsize=(10, 12))


    # Plotting the differences
    bars = ax.barh(indices, sorted_differences, color=palette)

    # Adding a vertical line at zero
    ax.axvline(0, color='black', linewidth=1)

    # Adding labels
    ax.set_xlabel('Difference in Mean Score (Reviewer - ChatGPT)')
    ax.set_ylabel('Query')
    #ax.set_title('Difference in Mean Scores (Reviewer - ChatGPT) with Stdev')
    ax.set_yticks(indices)
    ax.set_yticklabels([f'Q{i}' for i in sorted_indices], rotation=0)

    # Adding gridlines
    ax.xaxis.grid(True)
    ax.set_xticks([-2.25,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25])

    plt.tight_layout()
    plt.savefig("Reviewer_vs_ChatGPT_Difference_Scores_with_Stdev.png")
    plt.show()

# Call the function to plot the differences
plot_difference(df)


# Calculate the average for all Q1, Q2, Q3, etc. replicates for ChatGPT and Reviewer
chatgpt_means = [df[f'Q{i}_ChatGPT_Mean'].mean() for i in range(1, 6)]
reviewer_means = [df[f'Q{i}_TA_Mean'].mean() for i in range(1, 6)]

chatgpt_std = [df[f'Q{i}_ChatGPT_std'].mean() for i in range(1, 6)]
reviewer_std = [df[f'Q{i}_TA_std'].mean() for i in range(1, 6)]

# Perform paired t-tests for each question
p_values = []
for i in range(1, 6):
    chatgpt_scores = df[[f'Q{i}_ChatGPT_Replicate_{j}' for j in range(1, 4)]].values.flatten()
    reviewer_scores = df[[f'Q{i}_TA_Replicate_{j}' for j in range(1, 4)]].values.flatten()
    t_stat, p_val = wilcoxon(chatgpt_scores, reviewer_scores)
    p_values.append(p_val)

# Function to plot the average scores for ChatGPT and Reviewer side by side with error bars and significance annotations
def plot_Q_average_scores_with_error_and_significance(chatgpt_means, reviewer_means, chatgpt_std, reviewer_std, p_values):
    questions = [f'Q{i}' for i in range(1, 6)]
    x = np.arange(len(questions))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.set(style="whitegrid")
    palette = sns.color_palette("coolwarm", 8)

    bars1 = ax.bar(x - width/2, chatgpt_means, width, yerr=chatgpt_std, label='ChatGPT', color=palette[-1], capsize=5)
    bars2 = ax.bar(x + width/2, reviewer_means, width, yerr=reviewer_std, label='Reviewer', color=palette[0], capsize=5)

    # Adding labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Questions')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Scores by Question for ChatGPT and Reviewer')
    ax.set_xticks(x)
    ax.set_xticklabels(['Understood Intent','Proper Resource', 'Relavent Information', 'Sufficient Detail', 'Generally Correct'])

    # Adding significance annotations
    for i, (chatgpt_mean, reviewer_mean, p_val) in enumerate(zip(chatgpt_means, reviewer_means, p_values)):
        height = max(chatgpt_mean + chatgpt_std[i], reviewer_mean + reviewer_std[i])
        significance = ''
        if p_val < 0.001:
            significance = '***'
        elif p_val < 0.01:
            significance = '**'
        elif p_val < 0.05:
            significance = '*'
        
        ax.text(i, height + 0.05, significance, ha='center', va='bottom', color='black', fontsize=20)

    # Adding gridlines
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig("ChatGPT_vs_Reviewer_Average_Scores_with_Error_and_Significance.png")
    plt.show()

# Call the function to plot the average scores with error bars and significance annotations
plot_Q_average_scores_with_error_and_significance(chatgpt_means, reviewer_means, chatgpt_std, reviewer_std, p_values)




