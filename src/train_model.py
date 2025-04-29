# train_model.py

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def main():
    # Load tokenized data
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(
        train_df['tweet'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    test_encodings = tokenizer(
        test_df['tweet'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    train_labels = torch.tensor(train_df['label'].tolist())
    test_labels = torch.tensor(test_df['label'].tolist())

    # Prepare datasets
    train_dataset = HateSpeechDataset(train_encodings, train_labels)
    test_dataset = HateSpeechDataset(test_encodings, test_labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Prepare optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Prepare learning rate scheduler
    num_training_steps = len(train_dataset) // 16 * 3   # batch_size=16, epochs=3
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(3):  # 3 epochs
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} complete!")

    # Save model
    model.save_pretrained('../models/bert-hatespeech-new')
    tokenizer.save_pretrained('../models/bert-hatespeech-new')

    print("Training complete and model saved!")

if __name__ == "__main__":
    main()
