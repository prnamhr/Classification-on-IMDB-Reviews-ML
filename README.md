# IMDB Sentiment Classification (FastText, LSTM, BERT)

This project trains and compares three sentiment classifiers on the **IMDB movie reviews** dataset, where labels are:
- `1` = positive review
- `0` = negative review

## What’s Included
- **FastText** supervised classifier (with parameter tuning)
- **LSTM** model in Keras (Embedding → LSTM → Sigmoid)
- **BERT** fine-tuning using HuggingFace Transformers (DistilBERT)

## Quick Summary of Steps
1. Load IMDB data and decode token IDs back into text.
2. Create FastText input files:
   - `imdb_train.txt` (first 10,000 training examples)
   - `imdb_test.txt` (first 5,000 test examples)
   - Format per line: `__label__{y} {review_text}`
3. Train FastText with default settings, then tune `lr`, `epoch`, `wordNgrams`, and `dim`.
4. Train an LSTM model using padded sequences (`maxlen=200`) and compare embedding size + epochs.
5. Fine-tune `distilbert-base-uncased` for binary classification and evaluate accuracy for different epoch counts.

## Outputs
- FastText precision (default vs tuned)
- LSTM test accuracy for different embedding dimensions and epochs
- BERT evaluation accuracy (and loss) for different training epochs
