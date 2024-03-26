import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def pipe(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes a DataFrame by splitting it into training and testing sets, ensuring the training set is balanced, and then
    training and predicting with a text processing pipeline. This pipeline uses CountVectorizer, TfidfTransformer, and
    MultinomialNB for classification.

    Args:
        df (pd.DataFrame): Input DataFrame with 'text' and 'target' columns.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features (X_train), testing features (X_test),
        training labels (y_train), and testing labels (y_test), ready for model training and evaluation.
    """
    print("Creating Train, Test split...\n\n")
    X_train, X_test, y_train, y_test = get_train_test_split(df=df)

    print("Done! check for balance ... \n\n")
    plot_balance(y_train=y_train, y_test=y_test)

    print("Looks unbalanced! Balancing Train Set... \n\n")
    X_train, y_train = balance_train_set(X_train=X_train, y_train=y_train)

    plot_balance(y_train=y_train, y_test=y_test)
    print("Balanced! \n\n")

    # reshape back into (-1,)
    X_train = X_train.reshape(-1)
    y_train = y_train.reshape(-1)

    fit_and_predict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def get_train_test_split(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame to be split, containing the 'text' and 'target' columns.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the training features (X_train),
            testing features (X_test), training labels (y_train), and testing labels (y_test).
    """
    X = df["text"].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    return X_train, X_test, y_train, y_test


def plot_balance(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plots the balance of labels in the training and testing sets to visualize the distribution of classes.

    Args:
        y_train (np.ndarray): The labels of the training set.
        y_test (np.ndarray): The labels of the testing set.

    Returns:
        None: This function does not return a value but displays a bar plot showing the count of each label in the training and testing datasets.
    """
    # get value counts
    ones_y_train = np.count_nonzero(y_train == 1)
    zeros_y_train = np.count_nonzero(y_train == 0)
    ones_y_test = np.count_nonzero(y_test == 1)
    zeros_y_test = np.count_nonzero(y_test == 0)

    # prep data
    labels = ["Ones", "Zeros"]
    train_counts = [ones_y_train, zeros_y_train]
    test_counts = [ones_y_test, zeros_y_test]

    # label positions
    x = np.arange(len(labels))

    # bar width
    width = 0.35

    # create barplots
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, train_counts, width, label="Train")
    rects2 = ax.bar(x + width / 2, test_counts, width, label="Test")
    ax.set_ylabel("Count")
    ax.set_title("Balance in Train / Test Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """actual values above bars"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def balance_train_set(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balances the training set to ensure an equal number of instances for each class using Random Over Sampling.

    Args:
        X_train (np.ndarray): The features of the training set before balancing.
        y_train (np.ndarray): The labels of the training set before balancing.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the balanced training features (X_train) and labels (y_train).
    """
    ros = RandomOverSampler(random_state=42)

    X_training = X_train.reshape(-1, 1)
    y_training = y_train.reshape(-1, 1)

    X_train, y_train = ros.fit_resample(X_training, y_training)

    return X_train, y_train


def fit_and_predict(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Fits a text processing and classification pipeline on training data and makes predictions on test data. The pipeline
    includes CountVectorizer for tokenizing text data and converting it into a matrix of token counts, TfidfTransformer
    for computing term frequency-inverse document frequency to reflect the importance of words to a document, and
    MultinomialNB for Naive Bayes classification. Finally, prints a classification report comparing the predictions to
    the true labels in the test set.

    Args:
        X_train (np.ndarray): Training features, expected to be a numpy array of text data.
        X_test (np.ndarray): Test features, expected to be a numpy array of text data.
        y_train (np.ndarray): Training labels, expected to be a numpy array of target values.
        y_test (np.ndarray): Test labels, expected to be a numpy array of target values.
    """
    # build pipeline
    pipeline = Pipeline(
        [
            ("count_vectorizer", CountVectorizer()),
            ("tfidf_transformer", TfidfTransformer()),
            ("naive_bayes", MultinomialNB()),
        ]
    )

    print("Fitting Pipeline... \n\n")
    pipeline.fit(X_train, y_train)

    print("predicting... \n\n")
    y_pred = pipeline.predict(X_test)

    # print classification report
    print(classification_report(y_test, y_pred))
