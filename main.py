from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import train, parse_arguments


# ********************************************************************************************
# *                                                                                          *
# *                 Load and vectorize the 20 newsgroups dataset                              *
# *                                                                                          *
# ********************************************************************************************
def load_dataset(
    remove=(),
    params=None,
    is_TFIDF=False,
):
    """
    Load and vectorize the 20 newsgroups dataset.
    """

    if params is None:
        params = {
            "lowercase": True,
            "stop_words": "english",
            "analyzer": "word",
            "ngram_range": (1, 1),
            "max_features": None,
        }

    data_train = fetch_20newsgroups(
        subset="train",
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    data_test = fetch_20newsgroups(
        subset="test",
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    target_names = data_train.target_names
    y_train, y_test = data_train.target, data_test.target

    # Extracting features from the training and test data using a Tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.5,
        min_df=5,
        stop_words=params["stop_words"],
        lowercase=params["lowercase"],
        analyzer=params["analyzer"],
        ngram_range=params["ngram_range"],
        max_features=params["max_features"],
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(data_train.data)
    X_test_tfidf = tfidf_vectorizer.transform(data_test.data)
    if is_TFIDF:
        return X_train_tfidf, X_test_tfidf, y_train, y_test, target_names

    # Extracting features from the training and test data using a count vectorizer
    count_vectorizer = CountVectorizer(max_df=0.5, min_df=5, stop_words="english")
    X_train_count = count_vectorizer.fit_transform(data_train.data)
    X_test_count = count_vectorizer.transform(data_test.data)

    # Extracting features from the training and test data using a Tf vectorizer
    tf_vectorizer = TfidfVectorizer(
        sublinear_tf=True, use_idf=False, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train_tf = tf_vectorizer.fit_transform(data_train.data)
    X_test_tf = tf_vectorizer.transform(data_test.data)

    return (
        {
            "tfidf": (X_train_tfidf, X_test_tfidf),
            "tf": (X_train_tf, X_test_tf),
            "count": (X_train_count, X_test_count),
        },
        y_train,
        y_test,
        target_names,
    )


# ********************************************************************************************
# *                                                                                          *
# *                 Experiment with different classifiers and features                       *
# *                                                                                          *
# ********************************************************************************************
def compare_classifiers_features(classifiers):
    """
    This function compares different classifiers and features to find the best combination.
    """
    features, y_train, y_test, target_names = load_dataset(
        remove=("headers", "footers", "quotes")
    )

    for feature in features:
        X_train, X_test = features[feature]
        print(
            f"******************************* Feature: {feature} *******************************"
        )
        for clf in classifiers:
            train(
                clf,
                X_train,
                y_train,
                X_test,
                y_test,
                target_names,
                feature,
                file_name="clf_feature_results.csv",
            )


# ********************************************************************************************
# *                                                                                          *
# *        Experiment with different parameters with the best combination of the             *
# *        classifier and feature: ComplementNB with TfidfVectorizer                         *
# *                                                                                          *
# ********************************************************************************************
def experiment_params(params):
    """
    This function iterates over different values of lowercase, stop_words, analyzer, ngram_range, and max_features
    as specified in the params dictionary.

    Parameters:
        - params (dict): A dictionary that defines the values for various vectorization parameters, including:
        - "lowercase": Whether to convert text to lowercase.
        - "stop_words": Stop word removal options based on the analyzer type (word or char).
        - "analyzer": Determines whether to use word or character analysis.
        - "ngram_range_word": The range of n-grams for word analysis.
        - "ngram_range_char": The range of n-grams for character analysis.
        - "max_features": The maximum number of features to use in the vectorization.
    """
    for lowercase in params["lowercase"]:
        for analyzer in params["analyzer"]:
            stop_words_list = params["stop_words"][analyzer]
            for stop_words in stop_words_list:
                ngram_ranges = (
                    params["ngram_range_word"]
                    if analyzer == "word"
                    else params["ngram_range_char"]
                )
                for ngram_range in ngram_ranges:
                    for max_features in params["max_features"]:
                        print(
                            "params:",
                            lowercase,
                            stop_words,
                            analyzer,
                            ngram_range,
                            max_features,
                        )
                        X_train_tfidf, X_test_tfidf, y_train, y_test, target_names = (
                            load_dataset(
                                remove=("headers", "footers", "quotes"),
                                params={
                                    "lowercase": lowercase,
                                    "stop_words": stop_words,
                                    "analyzer": analyzer,
                                    "ngram_range": ngram_range,
                                    "max_features": max_features,
                                },
                                is_TFIDF=True,
                            )
                        )
                        train(
                            ComplementNB(),
                            X_train_tfidf,
                            y_train,
                            X_test_tfidf,
                            y_test,
                            target_names,
                            feature="TfidfVectorizer",
                            file_name="param_experiment_results.csv",
                            params={
                                "lowercase": lowercase,
                                "stop_words": stop_words,
                                "analyzer": analyzer,
                                "ngram_range": ngram_range,
                                "max_features": max_features,
                            },
                        )


if __name__ == "__main__":

    args = parse_arguments()

    if args.compare:
        compare_classifiers_features(
            classifiers=[
                RidgeClassifier(tol=1e-2, solver="sparse_cg"),
                SVC(kernel="linear"),
                RandomForestClassifier(),
                MultinomialNB(),
                ComplementNB(),
                LogisticRegression(),
            ]
        )
    # ********************************************************************************************
    # *                                                                                          *
    # *        Best combination of classifier and feature is ComplementNB with TfidfVectorizer    *
    # *                                                                                          *
    # ********************************************************************************************
    if args.experiment:
        experiment_params(
            params={
                "lowercase": [True, False],
                "stop_words": {"word": [None, "english"], "char": [None]},
                "analyzer": ["word", "char"],
                "ngram_range_word": [(1, 1), (1, 2), (2, 3)],
                "ngram_range_char": [(3, 5), (3, 7)],
                "max_features": [None, 1000, 2000, 5000],
            }
        )
