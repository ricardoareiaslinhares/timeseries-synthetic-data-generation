"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

data.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
(3) load_data: download or generate data
(4): batch_generator: mini-batch generator
"""

import numpy as np
import pandas as pd
from os.path import dirname, abspath


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ["stock", "energy"]

    if data_name == "stock":
        ori_data = np.loadtxt(
            dirname(dirname(abspath(__file__))) + "/data/stock_data.csv",
            delimiter=",",
            skiprows=1,
        )
    elif data_name == "energy":
        ori_data = np.loadtxt(
            dirname(dirname(abspath(__file__))) + "/data/energy_data.csv",
            delimiter=",",
            skiprows=1,
        )

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i : i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def load_data(opt):
    ## Data loading
    if opt.data_name in ["stock", "energy"]:
        ori_data = real_data_loading(opt.data_name, opt.seq_len)  # list: 3661; [24,6]
    elif opt.data_name == "sine":
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, opt.seq_len, dim)
    print(opt.data_name + " dataset is ready.")

    return ori_data


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def sequencing_data(data_df, seconds, sampling_rate, overlap=0.5, normalize=False):
    """_summary_
        Sequences a TS df into a nparray of mutiple sequences.
    Args:
        data_df (pd.DataFrame): _description_
        seconds (number): seconds to be sequenced (sec = seq_length / sampling_rate)
        sampling_rate (number): _description_
        overlap (float, optional): _description_. Percentage of the overlap between sequences. Defaults to 0.5
        normalize (bool, optional): applying minmax normalization. Defaults to False.

    Returns:
        np.array: A np.array with shape (n_samples, seq_length, n_features)
    """

    def MinMaxScaler(data):
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        return (data - min_vals) / (max_vals - min_vals + 1e-7)

    if normalize:
        data_df = MinMaxScaler(data_df)

    seq_length = int(seconds * sampling_rate)

    # Overlap
    step_size = int(
        seq_length * (1 - overlap)
    )  # For 50% overlap, overlap=0.5, step_size=seq_length // 2

    # Build dataset sequences
    dataX = []
    for i in range(0, len(data_df) - seq_length, step_size):
        _x = data_df.iloc[
            i : i + seq_length
        ].to_numpy()  # Convert only the current sequence
        dataX.append(_x)

    # Mix Data (shuffle to make i.i.d-like)
    idx = np.random.permutation(len(dataX))

    outputX = [dataX[i] for i in idx]

    # return np.array(outputX)
    return outputX


def load_data_2(timegan_settings):
    data = np.load(timegan_settings.data_name, allow_pickle=True)
    columns_names = timegan_settings.columns

    train_data = pd.DataFrame(data[timegan_settings.activity])

    if len(train_data.columns) > len(columns_names):
        train_data = train_data.iloc[:, : len(columns_names)]

    train_data.rename(
        columns={
            column: columns_names[idx] for idx, column in enumerate(train_data.columns)
        },
        inplace=True,
    )

    # segment data
    train_data_windows = sequencing_data(
        train_data,
        seconds=timegan_settings.seconds,
        sampling_rate=timegan_settings.sampling_rate,
        overlap=0.99,
        normalize=True,
    )
    return train_data_windows
