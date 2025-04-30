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


## Model Performance (after cleaning improvements)

| Metric | Score % |
|-------------|---------|
| `Accuracy` | 91.61 |
| `Precision` | 91.28 |
| `Recall` | 91.61 |
| `F1 Score` | 91.43 |


## Dataset Used
[Davidson Hate Speech and Offensive Language Dataset
](https://github.com/t-davidson/hate-speech-and-offensive-language)

~25,000 labeled tweets

Class distribution:

Hate Speech (5%)

Offensive Language (66%)

Neither (29%)


## Model Architecture

Pretrained: bert-base-uncased (Hugging Face Transformers)

Fine-tuned on cleaned tweet data

3-class classification head

3 epochs, batch size 16, AdamW optimizer





