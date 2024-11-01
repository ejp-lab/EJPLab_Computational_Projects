import pandas as pd
import hashlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from contact_chatgpt import ask_gpt 
from scipy.stats import chi2_contingency
import os
import json
from vectorize_documents import clean_and_embed
from sklearn.metrics import silhouette_score
import re
import collections
import pandas as pd
import collections
import seaborn as sns

# Load the CSV file
df = pd.read_csv('deidentified_dataset.csv')
grades_df = pd.read_csv("student_grades_with_codenames.csv")

def create_word_cloud(text_series, title, colors):
    # Combine all texts into one large string
    combined_text = " ".join(review for review in text_series.dropna())
    
    # Define a custom color map from the provided list of colors
    custom_color_map = LinearSegmentedColormap.from_list("custom_spectrum", colors)
    
    # Create and generate a word cloud image using the custom color map
    wordcloud = WordCloud(background_color='white', max_words=200, contour_width=3,
                          colormap=custom_color_map).generate(combined_text)
    
    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()

df['Body'] = df['Body'].str.replace('<FILTERED>', '', regex=False) # Removing Filtered
df = df[df['Sender'] != 'Person_325c7c08'] # Sends from icloud and has formatting issues

# Define color spectrums for each word cloud
cool_colors = ['teal', 'deepskyblue', 'blue', 'purple']
warm_colors = ['pink', 'red', 'darkred', 'maroon']

# Generate word clouds with the custom color spectrums
create_word_cloud(df['Response'], 'Word Cloud for Response Column', cool_colors)
create_word_cloud(df['Body'], 'Word Cloud for Body Column', warm_colors)

# Vectorize the 'Body' text
def optimal_clusters_elbow(embeddings, min_clusters=8, max_clusters=10):
    inertia = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
    
    # Return the optimal number of clusters
    optimal_n = min_clusters + np.argmin(np.gradient(inertia))
    return optimal_n

# Function to perform clustering with the optimal number of clusters
def perform_clustering(df, column_name, embeddings_file_path, file_name, min_clusters=8, max_clusters=10):
    """
    Perform clustering on a specified DataFrame column, using the elbow method to find the optimal number of clusters,
    and create a visualization of clusters.
    """
    # Load or compute embeddings
    if os.path.exists(embeddings_file_path):
        with open(embeddings_file_path, 'r') as f:
            embeddings = json.load(f)
    else:
        embeddings = [clean_and_embed(text) for text in df[column_name]]
        
        # Save the embeddings to the specified file
        with open(embeddings_file_path, 'w') as f:
            json.dump(embeddings, f)
    
    # Convert embeddings to a numpy array
    numeric_embeddings = np.array([e[0] for e in embeddings])
    
    # Get optimal clusters using the elbow method
    optimal_n_clusters = 9#optimal_clusters_elbow(numeric_embeddings, min_clusters, max_clusters)
    
    # Clustering using the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init="auto")
    kmeans.fit(numeric_embeddings)
    df[f"{column_name}_cluster"] = kmeans.labels_
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(numeric_embeddings)
    
    # Create a scatter plot to visualize clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df[f"{column_name}_cluster"], cmap="viridis", alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"Clustering of {column_name} Data (Optimal Clusters)")
    plt.colorbar(label="Cluster")
    plt.savefig(file_name)

    # Return the updated DataFrame with cluster information
    return df

# Perform clustering on the 'Body' column
df = perform_clustering(df, "Body", "student_embeddings.json", "student_clustering")
df = perform_clustering(df, 'Response', "response_embeddings.json", "response_clustering")

contingency_table = pd.crosstab(df['Body_cluster'], df['Response_cluster'])

# Visualize the contingency table as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='coolwarm')
plt.title("Contingency Table: Body_cluster vs Response_cluster")
plt.xlabel("Response Cluster")
plt.ylabel("Body Cluster")
plt.show()

# Perform a chi-squared test to check for significant associations
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-squared test statistic: {chi2}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p}")

# Determine if the result is statistically significant (commonly, p < 0.05)
if p < 0.05:
    print("There is a significant association between 'Body_cluster' and 'Response_cluster'.")
else:
    print("There is no significant association between 'Body_cluster' and 'Response_cluster'.")

def word_frequency(text_series):
    """
    Calculate word frequency in a given text series.
    :param text_series: A pandas Series containing text data.
    :return: A Counter object with word frequencies.
    """
    # Tokenize and normalize text
    words = []
    for text in text_series:
        # Convert to lowercase and remove non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        words.extend(tokens)

    # Count word frequency
    return collections.Counter(words)

def plot_word_frequency(counter, top_n=10, title="Word Frequency"):
    """
    Plot the word frequency for the top N words.
    :param counter: A Counter object with word frequencies.
    :param top_n: The number of top words to display.
    :param title: The title for the plot.
    """
    # Get the most common words
    common_words = counter.most_common(top_n)
    
    # Create a bar plot
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), orient="h")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.show()

# Calculate word frequency for each cluster
def analyze_cluster_word_frequency(df, column_name, cluster_column, top_n=10):
    """
    Analyze word frequency for each cluster and plot the most common words.
    :param df: The DataFrame containing the data.
    :param column_name: The column with the text data.
    :param cluster_column: The column with the cluster assignments.
    :param top_n: The number of top words to display.
    """
    unique_clusters = df[cluster_column].unique()

    for cluster in unique_clusters:
        cluster_data = df[df[cluster_column] == cluster]
        word_freq = word_frequency(cluster_data[column_name])
        
        # Plot the word frequency for the current cluster
        plot_word_frequency(word_freq, top_n, title=f"Word Frequency for {cluster_column} {cluster}")

# Analyze word frequency for 'Body_cluster'
#analyze_cluster_word_frequency(df, 'Body', 'Body_cluster', top_n=10)

# Analyze word frequency for 'Response_cluster'
#analyze_cluster_word_frequency(df, 'Response', 'Response_cluster', top_n=10)

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

usage_counts = df['Sender_Codename'].value_counts().reset_index()
usage_counts.columns = ['Codename', 'Usage_Frequency']

# Load the grades dataset and skip the first two data rows after the column titles
grades_df = pd.read_csv('student_grades_with_codenames.csv')

# Merge the usage frequency data with the grades data
merged_data = pd.merge(grades_df, usage_counts, on='Codename', how='left')

# Replace NaN in Usage_Frequency with 0 for students who did not use the service
merged_data['Usage_Frequency'].fillna(0, inplace=True)

# Remove rows where Codename is 'Unknown'
merged_data = merged_data[merged_data['Codename'] != 'Unknown']

# Save the merged data to a new CSV (optional)
merged_data.to_csv('merged_dataset_with_usage_and_grades.csv', index=False)

# Calculate the correlation between usage frequency and final score
correlation = merged_data['Usage_Frequency'].corr(merged_data['Final Score'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x='Usage_Frequency', y='Final Score')

# Fit a linear regression model
from sklearn.linear_model import LinearRegression
X = merged_data['Usage_Frequency'].values.reshape(-1, 1)
y = merged_data['Final Score'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = model.score(X, y)

# Plot the regression line
plt.plot(merged_data['Usage_Frequency'], y_pred, color='red', linewidth=2)

# Annotate the plot with the R^2 value
plt.text(0.05, 0.95, f'$R^2$ = {r2:.2f}', ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')

# Set plot labels and title
plt.xlabel('Usage Frequency')
plt.ylabel('Final Score')
plt.title('Correlation between Service Usage Frequency and Final Scores')

# Show the plot
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(grades_df['Final Score'], bins=20, kde=False)
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.title('Distribution of Final Scores')

# Show the histogram
plt.show()
