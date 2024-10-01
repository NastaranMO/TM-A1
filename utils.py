import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import csv
import os
import argparse


def parse_arguments():
    """
    Available arguments:

    - compare classifiers and features to find the best combination (--compare )
    - experiment with different parameters such as lowercase, stop_words, analyzer, ngram_range, and max_features (--experiment)

    """
    parser = argparse.ArgumentParser(
        description="20 Newsgroups Classification script for comparing classifiers and experimenting with different parameters."
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare classifiers and features to find the best combination",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Experiment with different parameters such as lowercase, stop_words, analyzer, ngram_range, and max_features",
    )
    return parser.parse_args()


def train(
    clf,
    X_train,
    y_train,
    X_test,
    y_test,
    target_names,
    feature,
    file_name="param_experiment_results.csv",
    params=None,
):
    """
    Train a classifier, make predictions and evaluate the performance.
    """
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # plot(y_test, pred, target_names, clf, feature_name=feature)
    print(f"Report performance of {clf.__class__.__name__}")
    acc_score = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted")
    recall = recall_score(y_test, pred, average="weighted")
    f1 = f1_score(y_test, pred, average="weighted")
    save_csv(
        clf.__class__.__name__,
        feature,
        acc_score,
        precision,
        recall,
        f1,
        csv_filename=file_name,
        params=params,
    )

    print("Precision Score", precision)
    print("Accuracy Score", acc_score)


def plot(y_test, pred, target_names, clf, feature_name, show=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.xaxis.set_ticklabels(target_names, rotation=45, ha="right")
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(f"Confusion Matrix for {clf.__class__.__name__}\n")
    plt.tight_layout()
    plt.savefig(
        f"./confusion_matrix/confusion_matrix_{clf.__class__.__name__} with {feature_name}.png"
    )
    if show:
        plt.show()
        plt.close()


def save_csv(
    clf_name,
    feature,
    accuracy,
    precision,
    recall,
    f1,
    csv_filename,
    params=None,
):
    """
    Save the classifier performance to a CSV file.
    """
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            if params:
                writer.writerow(
                    [
                        "lowercase",
                        "stop_words",
                        "analyzer",
                        "ngram_range",
                        "max_features",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                    ]
                )
            else:
                writer.writerow(
                    ["Classifier", "Feature", "Accuracy", "Precision", "Recall", "F1"]
                )

        # Write the classifier name, precision, and accuracy scores
        if params:
            writer.writerow(
                [
                    params["lowercase"],
                    params["stop_words"] if params["stop_words"] else "None",
                    params["analyzer"],
                    params["ngram_range"],
                    params["max_features"] if params["max_features"] else "None",
                    accuracy,
                    precision,
                    recall,
                    f1,
                ]
            )
        else:
            writer.writerow([clf_name, feature, accuracy, precision, recall, f1])
