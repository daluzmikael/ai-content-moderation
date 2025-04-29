# split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(csv_path):
    """
    Load the preprocessed CSV and split into train/test sets.
    """
    # Load
    df = pd.read_csv(csv_path)

    # Check if cleaned_tweet exists
    if 'cleaned_tweet' not in df.columns:
        raise ValueError("Data must have a 'cleaned_tweet' column. Please run preprocessing first.")

    # Split into features (X) and labels (y)
    X = df['cleaned_tweet']
    y = df['class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load preprocessed dataset
    df = pd.read_csv('../data/cleaned_data.csv')  # <<<< -------- load the NEW file

    # Check if cleaned_tweet exists
    if 'cleaned_tweet' not in df.columns:
        raise ValueError("Data must have a 'cleaned_tweet' column. Please run preprocessing first.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_tweet'],
        df['class'],
        test_size=0.2,
        random_state=42,
        stratify=df['class']
    )

    # Save splits
    train_df = pd.DataFrame({'tweet': X_train, 'label': y_train})
    test_df = pd.DataFrame({'tweet': X_test, 'label': y_test})

    train_df.to_csv('../data/train.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
