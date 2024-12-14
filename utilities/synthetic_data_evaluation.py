import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import pad_sequences
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def report_results(y_test, y_hat, class_names, save_path=""):
    """_summary_

    Args:
        y_test (_type_): _description_
        y_hat (_type_): _description_
        class_names ([]): Label names
        save_path (str, optional): _description_. Defaults to ''.

    Returns:
        Classification Report (pd.DataFrame).
        Confusion Matrix (plt.show).
        Acurracy Score (float): Can receive an optional (int) rounding argument.

    """

    def show_save_classification_report(
        y_test, y_hat, class_names, save_path="", rounding=3
    ):
        class_report = classification_report(y_test, y_hat, output_dict=True)
        df_class_report = pd.DataFrame(class_report).transpose()

        more_class_names = ["Accuracy", "Macro avg", "Weighted avg"]
        names_of_classes = class_names
        for name in more_class_names:
            if name not in names_of_classes:
                names_of_classes.append(name)

        df_class_report.index = names_of_classes
        df_class_report = df_class_report.round(rounding)
        df_class_report.style.set_caption("Classification Report")
        df_class_report.to_csv(save_path + "_class_report" + ".csv", index=True)

        return df_class_report

    def show_save_conf_matrix(y_test, y_hat, class_names, save_path=""):
        conf_matrix = confusion_matrix(y_test, y_hat)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 14},
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix", fontsize=16)
        plt.ylabel("True label", fontsize=14)
        plt.xlabel("Predicted label", fontsize=14)
        plt.savefig(save_path + "_conf_matrix" + ".png")
        plt.show()
        plt.close()

    def show_classification(rounding=3):
        classfication = show_save_classification_report(
            y_test, y_hat, class_names, save_path, rounding
        )
        return classfication

    def get_accuracy(rounding=0):
        acc = accuracy_score(y_test, y_hat)
        if rounding > 0:
            acc = round(acc, rounding)
        return acc

    def show_conf_matrix():
        show_save_conf_matrix(y_test, y_hat, class_names, save_path)

    return show_conf_matrix, show_classification, get_accuracy


def discriminative_score_metrics(dataX, dataX_hat):
    """_summary_

    Args:
        dataX (_type_): window of original data of one activity
        dataX_hat (_type_): window of synthetic data of the same activity

    Returns:
        _type_: _description_
    """
    # Clear any existing graph
    tf.keras.backend.clear_session()

    # Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0, :])

    # Compute Maximum seq length and each seq length
    dataT = [len(x[:, 0]) for x in dataX]
    Max_Seq_Len = max(dataT)

    # Network Parameters
    hidden_dim = max(int(data_dim / 2), 1)
    iterations = 2000
    batch_size = 128

    # Define the discriminator model
    def build_discriminator():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    hidden_dim, activation="tanh", return_sequences=False
                ),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    discriminator = build_discriminator()
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")

    # Train / Test Division
    def train_test_divide(dataX, dataX_hat, dataT):
        No = len(dataX)
        idx = np.random.permutation(No)
        train_idx = idx[: int(No * 0.8)]
        test_idx = idx[int(No * 0.8) :]

        trainX = [dataX[i] for i in train_idx]
        trainX_hat = [dataX_hat[i] for i in train_idx]
        testX = [dataX[i] for i in test_idx]
        testX_hat = [dataX_hat[i] for i in test_idx]

        trainT = [dataT[i] for i in train_idx]
        testT = [dataT[i] for i in test_idx]

        return trainX, trainX_hat, testX, testX_hat, trainT, testT

    # Pad data
    def pad_data(data, max_seq_len):
        return pad_sequences(data, maxlen=max_seq_len, padding="post", dtype="float32")

    # Prepare data
    trainX, trainX_hat, testX, testX_hat, trainT, testT = train_test_divide(
        dataX, dataX_hat, dataT
    )
    trainX = pad_data(trainX, Max_Seq_Len)
    trainX_hat = pad_data(trainX_hat, Max_Seq_Len)
    testX = pad_data(testX, Max_Seq_Len)
    testX_hat = pad_data(testX_hat, Max_Seq_Len)

    # Training step
    for itt in range(iterations):
        # Batch setting for real data
        idx = np.random.permutation(len(trainX))
        train_idx = idx[:batch_size]
        X_mb = np.array([trainX[i] for i in train_idx])

        # Batch setting for synthetic data
        idx = np.random.permutation(len(trainX_hat))
        train_idx = idx[:batch_size]
        X_hat_mb = np.array([trainX_hat[i] for i in train_idx])

        # Train discriminator
        real_loss = discriminator.train_on_batch(X_mb, np.ones((batch_size, 1)))
        fake_loss = discriminator.train_on_batch(X_hat_mb, np.zeros((batch_size, 1)))
        d_loss = 0.5 * (real_loss + fake_loss)

        if itt % 500 == 0:
            print(f"[step: {itt}] discriminator loss: {d_loss:.4f}")

    # Final Outputs (on Testing set)
    Y_pred_real_curr = discriminator.predict(np.array(testX))
    Y_pred_fake_curr = discriminator.predict(np.array(testX_hat))

    Y_pred_final = np.squeeze(
        np.concatenate((Y_pred_real_curr, Y_pred_fake_curr), axis=0)
    )
    Y_label_final = np.concatenate(
        (np.ones(len(Y_pred_real_curr)), np.zeros(len(Y_pred_fake_curr)))
    )

    # Accuracy
    Acc = accuracy_score(Y_label_final, Y_pred_final > 0.5)

    Disc_Score = np.abs(0.5 - Acc)

    return Disc_Score


def discriminative_score_metrics_matrix(real_matrix, synthetic_matrix):

    tf.keras.backend.clear_session()

    assert (
        real_matrix.shape[1] == synthetic_matrix.shape[1]
    ), "Input matrices must have the same number of features"

    data_dim = real_matrix.shape[1]

    hidden_dim = max(int(data_dim / 2), 1)
    iterations = 2000
    batch_size = 128

    # Define the discriminator model
    def build_discriminator():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim, activation="tanh", input_shape=(data_dim,)
                ),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    discriminator = build_discriminator()
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")

    # Train / Test Division
    def train_test_divide(real_data, synthetic_data):
        X = np.vstack((real_data, synthetic_data))
        y = np.concatenate((np.ones(len(real_data)), np.zeros(len(synthetic_data))))
        return train_test_split(X, y, test_size=0.2, shuffle=True)

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_divide(real_matrix, synthetic_matrix)

    # Training step
    for itt in range(iterations):
        # Batch setting
        idx = np.random.randint(0, len(X_train), batch_size)
        X_mb = X_train[idx]
        y_mb = y_train[idx]

        # Train discriminator
        d_loss = discriminator.train_on_batch(X_mb, y_mb)

        if itt % 500 == 0:
            print(f"[step: {itt}] discriminator loss: {d_loss:.4f}")

    # Final Outputs (on Testing set)
    Y_pred_final = discriminator.predict(X_test).squeeze()

    # Accuracy
    Acc = accuracy_score(y_test, Y_pred_final > 0.5)

    Disc_Score = np.abs(0.5 - Acc)

    return Disc_Score


def predictive_score_metrics(dataX, dataX_hat):
    """_summary_

    Args:
        dataX (_type_): Window data of one activity
        dataX_hat (_type_): window data of the same activity

    Returns:
        _type_: _description_
    """

    tf.keras.backend.clear_session()

    No = len(dataX)
    data_dim = len(dataX[0][0, :])

    dataT = [len(x[:, 0]) for x in dataX]
    Max_Seq_Len = max(dataT)

    hidden_dim = max(int(data_dim / 2), 1)
    iterations = 5000
    batch_size = 128

    # Define the predictor model
    def build_predictor():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    hidden_dim, activation="tanh", return_sequences=True
                ),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    predictor = build_predictor()
    predictor.compile(optimizer="adam", loss="mae")

    # Prepare data
    def prepare_data(data):
        X = [d[:-1, : (data_dim - 1)] for d in data]
        Y = [np.reshape(d[1:, (data_dim - 1)], [-1, 1]) for d in data]
        return X, Y

    X_hat, Y_hat = prepare_data(dataX_hat)
    X_real, Y_real = prepare_data(dataX)

    # Training using Synthetic dataset
    for itt in range(iterations):
        # Batch setting
        idx = np.random.permutation(len(X_hat))
        train_idx = idx[:batch_size]

        X_mb = np.array([X_hat[i] for i in train_idx])
        Y_mb = np.array([Y_hat[i] for i in train_idx])

        # Train predictor
        loss = predictor.train_on_batch(X_mb, Y_mb)

        if itt % 500 == 0:
            print(f"[step: {itt}] predictor loss: {loss:.4f}")

    # Use Original Dataset to test
    X_test = np.array(X_real)
    Y_test = np.array(Y_real)

    # Predict Future
    pred_Y = predictor.predict(X_test)

    # Compute MAE
    MAE_Temp = 0
    for i in range(No):
        MAE_Temp += mean_absolute_error(Y_test[i], pred_Y[i])

    MAE = MAE_Temp / No

    return MAE


def PCA_Analysis(dataX, dataX_hat):

    # Specifying Data Size
    Sample_No = 1000

    # Data Preprocessing
    for i in range(Sample_No):
        if i == 0:
            arrayX = np.reshape(
                np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])]
            )
            arrayX_hat = np.reshape(
                np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])]
            )
        else:
            arrayX = np.concatenate(
                (
                    arrayX,
                    np.reshape(
                        np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])]
                    ),
                )
            )
            arrayX_hat = np.concatenate(
                (
                    arrayX_hat,
                    np.reshape(
                        np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])]
                    ),
                )
            )

    # Parameters
    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]

    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(arrayX)
    pca_results = pca.transform(arrayX)
    pca_hat_results = pca.transform(arrayX_hat)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(
        pca_results[:, 0], pca_results[:, 1], c=colors[:No], alpha=0.2, label="Original"
    )
    plt.scatter(
        pca_hat_results[:, 0],
        pca_hat_results[:, 1],
        c=colors[No:],
        alpha=0.2,
        label="Synthetic",
    )

    ax.legend()

    plt.title("PCA plot")
    plt.xlabel("x-pca")
    plt.ylabel("y_pca")
    plt.show()


def tSNE_Analysis(dataX, dataX_hat, save_path="", size="md", alpha=0.4):

    # Specifying Data Size
    Sample_No = dataX.shape[0]

    # Preprocess
    for i in range(Sample_No):
        if i == 0:
            arrayX = np.reshape(
                np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])]
            )
            arrayX_hat = np.reshape(
                np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])]
            )
        else:
            arrayX = np.concatenate(
                (
                    arrayX,
                    np.reshape(
                        np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])]
                    ),
                )
            )
            arrayX_hat = np.concatenate(
                (
                    arrayX_hat,
                    np.reshape(
                        np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])]
                    ),
                )
            )

    # t-SNE Analysis together
    final_arrayX = np.concatenate((arrayX, arrayX_hat), axis=0)

    # Parameters
    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(final_arrayX)

    # Plotting
    size_fig = (
        (24, 20)
        if size == "lg"
        else (16, 12) if size == "md" else (12, 8) if size == "sm" else (8, 6)
    )
    figure, ax = plt.subplots(1, figsize=size_fig)

    plt.scatter(
        tsne_results[:No, 0],
        tsne_results[:No, 1],
        c=colors[:No],
        alpha=alpha,
        label="Original",
    )
    plt.scatter(
        tsne_results[No:, 0],
        tsne_results[No:, 1],
        c=colors[No:],
        alpha=alpha,
        label="Synthetic",
    )

    ax.legend()

    plt.title("t-SNE plot")
    plt.xlabel("x-tsne")
    plt.ylabel("y_tsne")
    plt.savefig(save_path + "tsne.png")
    plt.show()


def tSNE_Analysis_matrix(real_matrix, synthetic_matrix, save_path=""):
    # Ensure input matrices have the same number of features
    assert (
        real_matrix.shape[1] == synthetic_matrix.shape[1]
    ), "Input matrices must have the same number of features"

    # Combine real and synthetic data
    combined_data = np.vstack((real_matrix, synthetic_matrix))

    # Parameters
    n_real = real_matrix.shape[0]
    n_synthetic = synthetic_matrix.shape[0]
    colors = ["red"] * n_real + ["blue"] * n_synthetic

    # TSNE analysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(combined_data)

    # Plotting
    plt.figure(figsize=(10, 8))

    plt.scatter(
        tsne_results[:n_real, 0],
        tsne_results[:n_real, 1],
        c=colors[:n_real],
        alpha=0.2,
        label="Original",
    )
    plt.scatter(
        tsne_results[n_real:, 0],
        tsne_results[n_real:, 1],
        c=colors[n_real:],
        alpha=0.2,
        label="Synthetic",
    )

    plt.title("t-SNE plot")
    plt.xlabel("x-tsne")
    plt.ylabel("y-tsne")
    plt.legend()

    if save_path:
        plt.savefig("tsne_" + "save_path" + ".png")

    plt.show()
