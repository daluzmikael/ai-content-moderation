# AI Content Moderation Project

This project fine-tunes a BERT-based model to classify tweets as Hate Speech, Offensive Language, or Neither, based on the Davidson Hate Speech and Offensive Language dataset.

## Project Structure
- `src/` : Source code for preprocessing, training, evaluation
- `data/` : Cleaned and split datasets
- `models/` : Trained BERT models
- `plots/` : Confusion matrices

## How to Run
1. `python preprocess.py`
2. `python split_data.py`
3. `python train_model.py`
4. `python evaluate_model.py`

## Model Performance
- Accuracy: 91.61%
- Precision: 91.28%
- Recall: 91.61%
- F1 Score: 91.43%

## Ethical Reflection
We evaluated the model's potential biases and suggested improvements for fairness and reducing false positives.

---
