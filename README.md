# POS Tagging with GloVe and Logistic Regression

## Overview
This project trains a **Logistic Regression classifier** to perform **Part-of-Speech (POS) tagging** using **pre-trained GloVe word embeddings**. The model was trained on **the first 1000 sentences** from the [batterydata/pos_tagging dataset](https://huggingface.co/datasets/batterydata/pos_tagging) and tested on **the full dataset**. The final model achieved **84.2% accuracy**, exceeding the target of **70%**.

---

## Dataset
- **Source:** [Hugging Face dataset: batterydata/pos_tagging](https://huggingface.co/datasets/batterydata/pos_tagging)
- **Structure:** Each example contains:
  - `"words"`: A list of tokens in a sentence
  - `"labels"`: The corresponding POS tags for each token
- **Training Data:** **First 1000 sentences**
- **Test Data:** **All sentences in the dataset**

---

## Methodology

### 1. Load Pre-trained GloVe Embeddings
- Downloaded **GloVe 300d embeddings** from [Stanford NLP](https://nlp.stanford.edu/projects/glove/).
- Loaded embeddings into a dictionary for fast lookup.
- If a word was missing, assigned a **zero vector** as a fallback.

### 2. Extract Tokens and Build an Embedding Cache
- Retrieved **all unique words** from the dataset.
- Stored their **pre-trained GloVe vectors** in a dictionary.

### 3. Create Context-Aware Features
- Each token’s feature vector was **concatenated with its neighboring words** (previous and next token).
- Used a **context window of 1**, resulting in **900-dimensional feature vectors** (300d x 3).

### 4. Encode POS Tags into Numeric Labels
- Converted string POS tags (e.g., `"NN"`, `"VBZ"`) into numeric labels using `LabelEncoder()`.

### 5. Train a Logistic Regression Classifier
- Trained on the **first 1000 sentences**.
- Used **context-enhanced word embeddings** as features.

### 6. Evaluate Performance
- Measured **accuracy, precision, recall, and f1-score** on the test set.
