# evaluate_model.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Custom Dataset class (same as before)
class HateSpeechDataset(torch.utils.data.Dataset):
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
    # Load test data
    test_df = pd.read_csv('../data/test.csv')

    tokenizer = AutoTokenizer.from_pretrained('../models/bert-hatespeech-new')
    model = BertForSequenceClassification.from_pretrained('../models/bert-hatespeech-new')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    test_encodings = tokenizer(
        test_df['tweet'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    test_labels = torch.tensor(test_df['label'].tolist())
    test_dataset = HateSpeechDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('../plots/confusion_matrix_new.png')
    plt.show()

if __name__ == "__main__":
    main()
