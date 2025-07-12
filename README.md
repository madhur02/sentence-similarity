# Sentence Similarity Techniques Exploration

A comprehensive exploration of various sentence similarity techniques using different Natural Language Processing approaches. This project implements and compares multiple methods for measuring semantic similarity between sentences.

## Overview

This project explores four different approaches to sentence similarity:

1. **Universal Sentence Encoder (TensorFlow)** - Using Google's pre-trained transformer-based model
2. **Word2Vec-based Similarity** - Averaging word embeddings to represent sentences
3. **Doc2Vec-based Similarity** - Using document-level embeddings for sentence representation
4. **Stanford CoreNLP Parser** - Leveraging syntactic and semantic parsing features

## Project Structure

```
sentence-similarity-exploration/
├── README.md
├── requirements.txt
├── universal_sentence_encoder.py
├── word2vec_similarity.py
├── doc2vec_similarity.py
├── stanford_corenlp_similarity.py
├── data/
│   └── sample_sentences.txt
├── models/
│   └── (trained models will be saved here)
└── results/
    └── (similarity results and comparisons)
```

## Features

### 1. Universal Sentence Encoder
- Utilizes TensorFlow Hub's Universal Sentence Encoder
- Provides high-quality sentence embeddings
- Supports multiple languages
- Fast inference for real-time applications

### 2. Word2Vec Similarity
- Implements sentence similarity using Word2Vec word embeddings
- Averages word vectors to create sentence representations
- Handles out-of-vocabulary words gracefully
- Customizable pre-trained models (Google News, custom training)

### 3. Doc2Vec Similarity
- Uses Gensim's Doc2Vec implementation
- Learns distributed representations of sentences/documents
- Captures semantic relationships at document level
- Supports both training and inference modes

### 4. Stanford CoreNLP Similarity
- Leverages Stanford's CoreNLP toolkit
- Incorporates syntactic parsing and semantic analysis
- Provides detailed linguistic features
- Supports multiple similarity metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/madhur02/sentence-similarity-exploration.git
cd sentence-similarity-exploration
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. For Stanford CoreNLP, download the required models:
```bash
# Download Stanford CoreNLP
wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.0.zip
unzip stanford-corenlp-4.5.0.zip
```

## Usage

### Universal Sentence Encoder
```python
python universal_sentence_encoder.py
```

### Word2Vec Similarity
```python
python word2vec_similarity.py
```

### Doc2Vec Similarity
```python
python doc2vec_similarity.py
```

### Stanford CoreNLP Similarity
```python
python stanford_corenlp_similarity.py
```

## Example Usage

```python
# Example sentences
sentence1 = "The cat sat on the mat."
sentence2 = "A cat is sitting on a rug."

# Each module provides a similarity score between 0 and 1
# Higher scores indicate greater similarity
```

## Methodology

### Data Preprocessing
- Text cleaning and normalization
- Tokenization and lemmatization
- Handling of special characters and punctuation

### Similarity Metrics
- Cosine similarity for embedding-based methods
- Euclidean distance for geometric comparisons
- Custom metrics for parser-based approaches

### Evaluation
- Qualitative analysis of similarity scores
- Comparison across different techniques
- Performance benchmarking

## Results and Comparison

The project includes comprehensive analysis comparing:
- Accuracy of similarity detection
- Processing speed and efficiency
- Resource requirements
- Strengths and limitations of each approach

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Hub
- Gensim
- NLTK
- Stanford CoreNLP
- NumPy
- Pandas
- Scikit-learn
