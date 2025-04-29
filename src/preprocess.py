# src/preprocess.py

import re
import pandas as pd

def clean_text(text):
    text = text.lower()                                # Lowercase
    text = re.sub(r'http\S+', '', text)                 # Remove URLs
    text = re.sub(r'\b\d{5,}\b', '', text)              # Remove numbers with 5+ digits
    text = text.replace('&quot;', '"')                  # Decode HTML quotes
    text = text.replace('""', '"')                      # Fix double double-quotes
    text = re.sub(r'!{2,}', '!', text)                  # Replace 2+ '!' with one '!'
    text = re.sub(r'[^a-z0-9\s@\'"!]', '', text)         # <<< KEY FIX HERE!
    text = re.sub(r'\brt\b', '', text)                   # Remove standalone "rt"
    text = re.sub(r'\s+', ' ', text).strip()             # Normalize whitespace
    return text


def preprocess_dataframe(df):
    """
    Apply text cleaning to the 'tweet' column of the DataFrame.
    """
    df = df.copy()
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)
    return df

if __name__ == "__main__":
    # Load
    df = pd.read_csv('../data/labeled_data.csv')
    df = preprocess_dataframe(df)

    # Print a few sample rows
    print(df[['tweet', 'cleaned_tweet']].head())

    # Save new cleaned dataset
    df.to_csv('../data/cleaned_data.csv', index=False)
    print("Cleaned data saved to ../data/cleaned_data.csv")

    
