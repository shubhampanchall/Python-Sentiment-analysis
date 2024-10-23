import pandas as pd 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk 
from flask import Flask, request, jsonify, send_file 
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting 
import matplotlib.pyplot as plt 
from pymongo import MongoClient  # To work with MongoDB

# Initialize Flask app
app = Flask(__name__)

# Download VADER Lexicon (only required once)
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# MongoDB Connection
client = MongoClient('mongodb+srv://parth:2iz77vWGl6UbtGqx@itsm.cubqw.mongodb.net/test?retryWrites=true&w=majority&appName=itsm')
db = client['test']  # Database name
collection = db['T3B']  # Collection name

# Define the sentiment analysis function
def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Compound score reflects overall sentiment

# Define a new function to label based on the rating and sentiment
def assign_sentiment_label(row):
    if row['Overall_Satisfaction_ops'] <= 7:  
        return 'Negative'
    elif row['Sentiment'] > 0:
        return 'Positive'
    else:
        return 'Neutral'

# Function to determine if a negative comment is a product or service issue
def assign_issue_type(comment):
    product_keywords = ['broken', 'defective', 'quality', 'fault', 'malfunction', 'product']
    service_keywords = ['late','Technician''customer support', 'service', 'staff', 'experience', 'help']
    
    if any(keyword in comment.lower() for keyword in product_keywords):
        return 'Product Issue'
    if any(keyword in comment.lower() for keyword in service_keywords):
        return 'Service Issue'
    
    return 'Other'

# API route to perform sentiment analysis and save to MongoDB
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        # Load the file from request
        file = request.files['file']
        
        # Load data into a DataFrame
        df = pd.read_excel(file)

        # Clean Data (remove rows where the comment is NaN)
        df.dropna(subset=['Overall_Satisfaction_Comment'], inplace=True)

        # Apply sentiment analysis to the comments
        df['Sentiment'] = df['Overall_Satisfaction_Comment'].apply(get_vader_sentiment)

        # Apply the sentiment label logic
        df['Sentiment_Label'] = df.apply(assign_sentiment_label, axis=1)

        # For negative comments, assign issue type based on the comment text
        df['Issue_Type'] = df.apply(lambda row: assign_issue_type(row['Overall_Satisfaction_Comment']) if row['Sentiment_Label'] == 'Negative' else 'N/A', axis=1)

        # Save the updated DataFrame to an Excel file
        output_file_path = 'output_sentiment_analysis.xlsx'
        df.to_excel(output_file_path, index=False)

        # Plotting the sentiment distribution
        plt.figure(figsize=(8, 5))
        df['Sentiment_Label'].value_counts().plot(kind='bar', color=['blue', 'red', 'green'])
        plt.title('Sentiment Analysis Results')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(axis='y')

        # Save the plot to a file
        plot_file_path = 'sentiment_analysis_plot.png'
        plt.tight_layout()
        plt.savefig(plot_file_path)

        # Save analysis results to MongoDB
        records = df.to_dict(orient='records')
        inserted_ids = collection.insert_many(records)
        print(f"Inserted IDs: {inserted_ids.inserted_ids}")  # Print inserted IDs

        # Return success message with paths
        return jsonify({
            'message': 'Sentiment analysis completed successfully and data saved to MongoDB!',
            'output_excel': request.host_url + 'files/output_sentiment_analysis.xlsx',
            'output_plot': request.host_url + 'files/sentiment_analysis_plot.png'
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error message for debugging
        return jsonify({'error': str(e)}), 500


# Serve output files
@app.route('/files/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_file(filename, as_attachment=True)

# Main entry point for the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
