# 20 Newsgroups Classification

In this Text Mining Assignment we perform text classification on the 20 Newsgroups dataset using different classifiers and feature extraction techniques. 

## Preprocessing

In the preprocessing step, we tokenized the raw text, converted it to lowercase, and removed common English stop words to reduce noise. We applied different n-gram ranges to capture both individual words and word sequences, using both word-level and character-level tokenization. Additionally, we explored the impact of limiting the number of features using the `max_features` parameter in `TfidfVectorizer` to assess its effect on model performance.

## Features

1. **Classifiers**: RidgeClassifier, SVC, RandomForestClassifier, MultinomialNB, ComplementNB, LogisticRegression.
2. **Vectorizers**: CountVectorizer, Term Frequency (TF), TF-IDF.
3. **Parameter Tuning**: Lowercase conversion, stop words removal, analyzer type (word/char), n-gram ranges, and maximum feature limit.
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.
5. **CSV Output**: The results are stored in a CSV file (`clf_feature_results.csv` or `param_experiment_results.csv`) for different options.

## Dataset

The project uses the 20 Newsgroups dataset, which is a collection of approximately 20,000 newsgroup documents, organized into 20 different categories. The dataset can be fetched using `sklearn.datasets.fetch_20newsgroups`.

## Installation

You'll need to have Python installed along with the following dependencies:

- scikit-learn
- matplotlib

## Evaluation

Model performance is evaluated by:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

## Usage

This project contains two main functionalities:

1. **Comparing classifiers**: Compare different classifiers on various feature extraction techniques.
2. **Experimenting with parameters**: Experiment with various vectorization parameters (lowercase, stop words, analyzer, n-gram ranges, etc.) for the best classifier-feature combination.

### Running the script

Run the project by using the following commands:

1. **Compare Classifiers and Features**:
   
Run the following command to compare multiple classifiers (e.g., RidgeClassifier, SVC, etc.) across different feature extraction techniques (TF-IDF, CountVectorizer, etc.).

```bash
python main.py --compare
```

This will generate performance metrics for each combination of classifier and feature extraction technique, saved in the `clf_feature_results.csv` file.

2. **Experiment with Parameters**:
   
To experiment with different vectorization parameters using the best combination (ComplementNB with TF-IDF), run the following command:

```bash
python main.py --experiment
```

This functionality will test various configurations of lowercase conversion, stop words removal, analyzer type, n-gram ranges, and maximum feature limits, and save the results in `param_experiment_results.csv`.

## Code Overview

- `main.py`
  - `load_dataset()`: Load the dataset of 20newsgroup and vectorize it.
  - `compare_classifiers_features()`: Compares different classifiers with different features (TF-IDF, CountVectorizer, etc.) finally save it as CSV file in `clf_feature_results.csv`.
  - `experiment_params()`: Runs experiments with different vectorization parameters for the best classifier-feature combination and save the result in the `param_experiment_results`.
  
- `utils.py`
  - `parse_arguments()`
  - `train()`: Trains a classifier and evaluates the performance.
  - `plot()`: Generates a confusion matrix.
  - `save_csv()`

