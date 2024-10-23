import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

# Download VADER Lexicon (only required once)
nltk.download('vader_lexicon')

# Load your data (adjust the file path as necessary)
df = pd.read_excel('C:\\Users\\Dell\\OneDrive - techjar.in\\Desktop\\python sentiment\\T3B-feedback.xlsx')

# Print the columns to confirm the names
print("Columns in the DataFrame:")
print(df.columns)  # This will print the columns in your dataset

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Step 1: Clean Data (remove rows where the comment is NaN)
df.dropna(subset=['Overall_Satisfaction_Comment'], inplace=True)

# Step 2: Define the sentiment analysis function using VADER
def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score reflects overall sentiment

# Step 3: Apply sentiment analysis to the comments
df['Sentiment'] = df['Overall_Satisfaction_Comment'].apply(get_vader_sentiment)

# Step 4: Define a new function to label based on the rating and sentiment
def assign_sentiment_label(row):
    # Use the correct name of the rating column
    if row['Overall_Satisfaction_ops'] <= 7:  # Updated to use the correct rating column
        return 'Negative'
    else:
        return 'Positive' if row['Sentiment'] > 0 else 'Negative'

# Step 5: Apply the sentiment label logic
df['Sentiment_Label'] = df.apply(assign_sentiment_label, axis=1)

# Step 6: Save the updated DataFrame to a new Excel file
output_file_path = 'C:\\Users\\Dell\\OneDrive - techjar.in\\Desktop\\python sentiment\\updates_sentiment_analysis.xlsx'
df.to_excel(output_file_path, index=False)

# Plotting the sentiment distribution
plt.figure(figsize=(8, 5))
df['Sentiment_Label'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Sentiment Analysis Results')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.grid(axis='y')

# Save the plot to a file
plot_file_path = 'C:\\Users\\Dell\\OneDrive - techjar.in\\Desktop\\python sentiment\\sentiment_analysis_plot.png'
plt.tight_layout()
plt.savefig(plot_file_path)

# Confirmation message 
print(f"Sentiment analysis results have been saved to '{output_file_path}'.")
print(f"The plot has been saved to '{plot_file_path}'.")
