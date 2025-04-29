# prepare_for_bert.py

import pandas as pd
from transformers import AutoTokenizer

def tokenize_data(texts, tokenizer, max_length=128):
    """
    Tokenizes a list of texts using the provided tokenizer.
    Returns a dictionary of input_ids and attention_masks.
    """
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

if __name__ == "__main__":
    # Load train and test sets
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    # Load a pre-trained BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the tweets
    train_encodings = tokenize_data(train_df['tweet'], tokenizer)
    test_encodings = tokenize_data(test_df['tweet'], tokenizer)

    # Save tokenized data (optional: or pass them directly to model later)
    print(f"Train encodings: {train_encodings.keys()}")
    print(f"Test encodings: {test_encodings.keys()}")

