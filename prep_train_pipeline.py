import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


def prep_train_pipeline(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the training pipeline by splitting the data into train and test sets, checking for balance,
    balancing the train set if necessary, and reshaping the datasets.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features and target variable.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the reshaped training features (X_train),
        testing features (X_test), training labels (y_train), and testing labels (y_test).
    """
    print("Creating Train, Test split...")
    X_train, X_test, y_train, y_test = get_train_test_split(df=df)

    print("Done! check for balance ...")
    plot_balance(y_train=y_train, y_test=y_test)

    print("Looks unbalanced! Balancing Train Set...")
    X_train, y_train = balance_train_set(X_train=X_train, y_train=y_train)

    plot_balance(y_train=y_train, y_test=y_test)
    print("Done!")

    # reshape back into (-1,)
    X_train = X_train.reshape(-1)
    y_train = y_train.reshape(-1)

    return X_train, X_test, y_train, y_test


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
