# AI Content Moderation Using BERT

Using the Davidson Hate Speech and Offensive Language dataset, this project fine-tunes a BERT-based model to classify tweets as **Hate Speech**, **Offensive Language**, or **Neither**.  
The project demonstrates proper data preprocessing, model training, performance evaluation, and ethical reflection around bias in automated content moderation systems.

---

## Project Structure

| Folder/File | Purpose |
|-------------|---------|
| `src/` | All Python scripts for each stage of the project (modularized) |
| `data/` | Cleaned and split datasets (train/test) |
| `models/` | Fine-tuned BERT models and tokenizers |
| `plots/` | Confusion matrix visualizations |
| `notebooks/` | Jupyter notebook version of full pipeline (optional) |
| `README.md` | Project overview and documentation |

---

## How to Reproduce

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   
2. Run each script in order
   ```bash
   cd src
   python preprocess.py        # Clean and format the tweet data
   python split_data.py        # Split into training/testing sets
   python train_model.py       # Fine-tune BERT on training data
   python evaluate_model.py    # Evaluate model and generate metrics + confusion matrix

---

## Requirements

transformers

torch

scikit-learn

pandas

matplotlib

seaborn

notebook

---

## Model Performance (after cleaning improvements)

| Metric | Score % |
|-------------|---------|
| `Accuracy` | 91.61 |
| `Precision` | 91.28 |
| `Recall` | 91.61 |
| `F1 Score` | 91.43 |

---

## Dataset Used
[Davidson Hate Speech and Offensive Language Dataset
](https://github.com/t-davidson/hate-speech-and-offensive-language)

~25,000 labeled tweets

Class distribution:

Hate Speech (5%)

Offensive Language (66%)

Neither (29%)

---

## Model Architecture

Pretrained: bert-base-uncased (Hugging Face Transformers)

Fine-tuned on cleaned tweet data

3-class classification head

3 epochs, batch size 16, AdamW optimizer

---

## Preprocessing Details
### Removed:

URLs

Retweet markers (RT)

Emoji unicode numbers (e.g., 128128)

### Preserved:

@mentions to reflect targeting behavior

" quotes to retain quoted speech

Exclamation marks (!) for tone detection

----------------


# Ethical Reflection
## We considered ethical risks including:

Potential bias learned from imbalanced language targeting specific groups

False positives unfairly silencing benign content

False negatives missing truly harmful tweets

## Solutions proposed:

Use balanced datasets

Apply adversarial examples

Use threshold tuning

Implement human moderation for edge cases

---


# BERT

## About BERT

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based language model developed by Google in 2018. It was designed to understand the context of words based on their surroundings, which makes it especially effective for nuanced tasks like content moderation, sentiment analysis, and hate speech detection.

### Training
BERT is pretrained on a massive corpus before you fine-tune it. It learns deep language patterns through:

 Masked Language Modeling (MLM)
 
→ Random words are masked in a sentence, and BERT learns to predict them using the context around them (both left and right).
Example:

`"The cat sat on the [MASK]." → "mat"`

Next Sentence Prediction (NSP)

→ BERT is given two sentences and learns to predict whether the second one logically follows the first.
This helps with understanding relationships between ideas and sentences.

This pretraining was done on:

→ English Wikipedia (~2.5B words)

→ BookCorpus (800M words of fiction)

So before we ever fine-tuned it, BERT already understood English grammar, word relationships, and general world knowledge.

### Fine-Tuning in This Project

In our project, we took the bert-base-uncased version, which has:

12 Transformer layers

110 million parameters

WordPiece tokenizer (lowercased input, 30k vocab size)

We fine-tuned it on ~25,000 labeled tweets to adapt it specifically to hate/offensive language detection.



## Our usage of BERT

We fine-tuned a `bert-base-uncased` model to classify tweets into three categories:
Hate Speech (0), Offensive Language (1), and Neither (2).

### Hugging Face link:


[Hugging Face BERT
]([https://github.com/t-davidson/hate-speech-and-offensive-language](https://huggingface.co/docs/transformers/en/model_doc/bert
)

### To use the model:

`from transformers import AutoTokenizer, BertForSequenceClassification`

`tokenizer = AutoTokenizer.from_pretrained("your-username/bert-hatespeech")`
`model = BertForSequenceClassification.from_pretrained("your-username/bert-hatespeech")`

### Files include

| File | Purpose |
|-------------|---------|
| `pytorch_model.bin` | Model weights |
| `config.json` | Model architecture |
| `tokenizer.json` | Tokenizer used for input processing |
| `vocab.txt` | Vocabulary |
| `nspecial_tokens_map.json` | Special token definitions |


