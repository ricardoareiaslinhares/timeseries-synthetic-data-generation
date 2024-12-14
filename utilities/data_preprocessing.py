import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch


def separate_per_column(data_df, column_name, drop_column=True):
    """_summary_
    Receives a df and a column name, and returns a dictionary with the different values of the column name as keys, and respective dfs as values.
    Also drops the column specified (by default).
    Args:
        data_df (pd.DataFrame): _description_
        column_name (str): _description_
        drop_column (bool, optional): _description_. Defaults to True.

    Returns:
        dict: Returns a dictionary with the different values of the column name as keys, and respective dfs as values.
        Also drops the column specified.
    """
    dict_df = {}
    for value in data_df[column_name].unique():
        new_df = data_df[data_df[column_name] == value].copy()
        new_df.drop(columns=[column_name], inplace=True)
        dict_df[value] = new_df
    return dict_df


def downsample(data_df, current_sampling_rate, desired_sampling_rate):
    """_summary_
        Receives a df of a TS, it's sampling rate and a desired new sampling rate.
        Returns a new df with the data (down)sampled to the desired sampling rate.
    Args:
        data_df (pd.DataFrame): _description_
        current_sampling_rate (number): _description_
        desired_sampling_rate (number): _description_

    Returns:
        pd.DataFrame: A new df with the downsampled data
    """
    data_np = np.array(data_df)

    # Calculates desired number of samples
    n_samples = data_np.shape[0]
    new_n_samples = int(n_samples * (desired_sampling_rate / current_sampling_rate))

    # This fn transforms data to frequency domain, deletes to the number of samples, and convets back to time domain
    data_downsampled = signal.resample(data_np, new_n_samples)
    data_downsampled_df = pd.DataFrame(data_downsampled, columns=data_df.columns)
    return data_downsampled_df


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

    return np.array(outputX)


def sequencing_data_by_one(data_df, seconds, sampling_rate, normalize=False):

    def MinMaxScaler(data):
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        return (data - min_vals) / (max_vals - min_vals + 1e-7)

    if normalize:
        data_df = MinMaxScaler(data_df)

    seq_length = int(seconds * sampling_rate)

    # Build dataset sequences
    dataX = []
    for i in range(0, len(data_df) - seq_length):
        _x = data_df.iloc[i : i + seq_length]
        dataX.append(_x)

    # Mix Data (shuffle to make i.i.d-like)
    idx = np.random.permutation(len(dataX))

    outputX = [dataX[i] for i in idx]

    return np.array(outputX)


### FEATURES ###
################


def preselected_features(window, columns_names, sampling_rate=20):
    """_summary_
    Receives a window and returns a df with the computed features
    Args:
        window (_type_): a np.array with shape (seq_length, n_features).
        columns_names (List[str]): Requires to have features named like "x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro". Provide a list with the ordered features names.
        sampling_rate (int, optional): _description_. Defaults to 20.

    Returns:
        pd.DataFrame: A df of a single row with the computed features
    """

    accel_gyro = ["x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro"]
    accel_only = ["x-accel", "y-accel", "z-accel"]
    gyro_only = ["x-gyro", "y-gyro", "z-gyro"]
    features_column = (
        accel_gyro
        if "x-accel" in columns_names and "x-gyro" in columns_names
        else accel_only if "x-accel" in columns_names else gyro_only
    )

    features = {}
    window_df = pd.DataFrame(window, columns=features_column)

    def time_between_peaks(data):
        peaks, _ = find_peaks(data)
        if len(peaks) > 1:
            return np.diff(peaks).mean() * (1000 / sampling_rate)
        return np.nan

    def avg_absolute_deviation(data):
        return np.mean(np.abs(data - np.mean(data)))

    # resultant
    x2a, y2a, z2a, x2g, y2g, z2g = 0, 0, 0, 0, 0, 0
    for column in window_df.columns:
        if "x-accel" in column:
            x2a = window_df[column] ** 2 if column == "x-accel" else x2a
            y2a = window_df[column] ** 2 if column == "y-accel" else y2a
            z2a = window_df[column] ** 2 if column == "z-accel" else z2a
        if "x-gyro" in column:
            x2g = window_df[column] ** 2 if column == "x-gyro" else x2g
            y2g = window_df[column] ** 2 if column == "y-gyro" else y2g
            z2g = window_df[column] ** 2 if column == "z-gyro" else z2g

    if "x-accel" in window_df.columns:
        resultant_accel = np.sqrt(x2a + y2a + z2a)
    if "x-gyro" in window_df.columns:
        resultant_gyro = np.sqrt(x2g + y2g + z2g)

    for axis in window_df.columns:
        data = window_df[axis]
        features[f"mean_{axis}"] = data.mean()
        features[f"peak_{axis}"] = time_between_peaks(data)
        features[f"abs_dev_{axis}"] = avg_absolute_deviation(data)
        features[f"std_{axis}"] = np.std(data)

    if "x-accel" in window_df.columns:
        features["resultant_accel"] = resultant_accel.mean()
    if "x-gyro" in window_df.columns:
        features["resultant_gyro"] = resultant_gyro.mean()

    return pd.DataFrame(features, index=[0])


def preselected_features_by_a15(window, columns_names, sampling_rate=20, bins=10):
    """_summary_
    Receives a window and returns a df with the computed features
    Args:
        window (_type_): a np.array with shape (seq_length, n_features).
        columns_names (List[str]): Requires to have features named like "x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro". Provide a list with the ordered features names.
        sampling_rate (int, optional): _description_. Defaults to 20.

    Returns:
        pd.DataFrame: A df of a single row with the computed features
    """

    accel_gyro = ["x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro"]
    accel_only = ["x-accel", "y-accel", "z-accel"]
    gyro_only = ["x-gyro", "y-gyro", "z-gyro"]
    features_column = (
        accel_gyro
        if "x-accel" in columns_names and "x-gyro" in columns_names
        else accel_only if "x-accel" in columns_names else gyro_only
    )

    features = {}
    window_df = pd.DataFrame(window, columns=features_column)

    def time_between_peaks(data):
        peaks, _ = find_peaks(data)
        if len(peaks) > 1:
            return np.diff(peaks).mean() * (1000 / sampling_rate)
        return np.nan

    def avg_absolute_deviation(data):
        return np.mean(np.abs(data - np.mean(data)))

    # resultant
    x2a, y2a, z2a, x2g, y2g, z2g = 0, 0, 0, 0, 0, 0
    for column in window_df.columns:
        if "x-accel" in column:
            x2a = window_df[column] ** 2 if column == "x-accel" else x2a
            y2a = window_df[column] ** 2 if column == "y-accel" else y2a
            z2a = window_df[column] ** 2 if column == "z-accel" else z2a
        if "x-gyro" in column:
            x2g = window_df[column] ** 2 if column == "x-gyro" else x2g
            y2g = window_df[column] ** 2 if column == "y-gyro" else y2g
            z2g = window_df[column] ** 2 if column == "z-gyro" else z2g

    if "x-accel" in window_df.columns:
        resultant_accel = np.sqrt(x2a + y2a + z2a)
    if "x-gyro" in window_df.columns:
        resultant_gyro = np.sqrt(x2g + y2g + z2g)

    if "x-accel" in window_df.columns:
        features["resultant_accel"] = resultant_accel.mean()
    if "x-gyro" in window_df.columns:
        features["resultant_gyro"] = resultant_gyro.mean()

    for axis in window_df.columns:
        data = window_df[axis]

        features[f"mean_{axis}"] = data.mean()  # ok
        features[f"peak_{axis}"] = time_between_peaks(data)  # ok
        # features[f"abs_dev_{axis}"] = avg_absolute_deviation(data)
        features[f"std_{axis}"] = np.std(data)  # ok
        features[f"{axis}_aad"] = data.apply(
            lambda x: np.abs(x - data.mean())
        ).mean()  # ok
        hist, _ = np.histogram(data, bins=bins, density=True)
        for i in range(len(hist)):
            features[f"{axis}_bin_{i+1}"] = hist[i]

    return pd.DataFrame(features, index=[0])


def time_domain_features(window, columns_names, sampling_rate):
    features = {}

    accel_gyro = ["x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro"]
    accel_only = ["x-accel", "y-accel", "z-accel"]
    gyro_only = ["x-gyro", "y-gyro", "z-gyro"]
    features_column = (
        accel_gyro
        if "x-accel" in columns_names and "x-gyro" in columns_names
        else accel_only if "x-accel" in columns_names else gyro_only
    )

    window_df = pd.DataFrame(window, columns=features_column)

    for axis in features_column:
        data = window_df[axis]

        # Mean
        features[f"mean_{axis}"] = data.mean()

        # Median
        features[f"median_{axis}"] = data.median()

        # Min and its index
        features[f"min_{axis}"] = data.min()
        features[f"min_idx_{axis}"] = data.idxmin()

        # Max and its index
        features[f"max_{axis}"] = data.max()
        features[f"max_idx_{axis}"] = data.idxmax()

        # Range (Max - Min)
        features[f"range_{axis}"] = data.max() - data.min()

        # Root Mean Square (RMS)
        features[f"rms_{axis}"] = np.sqrt(np.mean(data**2))

        # Interquartile Range (IQR)
        features[f"iqr_{axis}"] = np.percentile(data, 75) - np.percentile(data, 25)

        # Mean Absolute Deviation (MAD)
        features[f"mad_{axis}"] = np.mean(np.abs(data - data.mean()))

        # Skewness
        features[f"skewness_{axis}"] = skew(data)

        # Kurtosis
        features[f"kurtosis_{axis}"] = kurtosis(data)

        # Entropy (using histogram approximation)
        histogram, bin_edges = np.histogram(data, bins=10, density=True)
        features[f"entropy_{axis}"] = entropy(histogram)

        # Energy (sum of squared values)
        features[f"energy_{axis}"] = np.sum(data**2)

        # Power (mean of squared values)
        features[f"power_{axis}"] = np.mean(data**2)

        # Harmonic Mean (uses scipy.stats)
        features[f"harmonic_mean_{axis}"] = stats.hmean(
            np.abs(data) + 1e-10
        )  # Avoid division by zero

    return pd.DataFrame(features, index=[0])


# Functions for frequency-domain features


def compute_psd(signal, sampling_rate):
    """Compute Power Spectral Density (PSD) using Welch's method."""
    nperseg = min(256, len(signal))
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=nperseg)
    return f, Pxx


def band_power(signal, sampling_rate, band):
    """Compute power within a specific frequency band."""
    f, Pxx = compute_psd(signal, sampling_rate)
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx_band], f[idx_band])


def dominant_frequency(signal, sampling_rate):
    """Find the dominant frequency and its magnitude."""
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    idx = np.argmax(np.abs(yf))
    return xf[idx], np.abs(yf[idx])


def mean_frequency(signal, sampling_rate):
    """Compute the mean frequency weighted by power."""
    f, Pxx = compute_psd(signal, sampling_rate)
    return np.sum(f * Pxx) / np.sum(Pxx)


def spectral_entropy(signal, sampling_rate):
    """Compute spectral entropy of the signal."""
    f, Pxx = compute_psd(signal, sampling_rate)
    Pxx_norm = Pxx / np.sum(Pxx)  # Normalize power spectrum
    return entropy(Pxx_norm)


def zero_crossings(signal):
    """Count the number of zero-crossings in the signal."""
    return np.sum(np.abs(np.diff(np.sign(signal))) > 0)


def frequency_domain_features(window, columns_names, sampling_rate, band):

    features = {}
    accel_gyro = ["x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro"]
    accel_only = ["x-accel", "y-accel", "z-accel"]
    gyro_only = ["x-gyro", "y-gyro", "z-gyro"]
    features_column = (
        accel_gyro
        if "x-accel" in columns_names and "x-gyro" in columns_names
        else accel_only if "x-accel" in columns_names else gyro_only
    )

    window_df = pd.DataFrame(window, columns=features_column)

    for axis in features_column:
        signal = window_df[axis].values

        features[f"{axis}_mean_frequency"] = mean_frequency(signal, sampling_rate)
        features[f"{axis}_band_power"] = band_power(signal, sampling_rate, band)
        features[f"{axis}_dominant_freq"], features[f"{axis}_dominant_mag"] = (
            dominant_frequency(signal, sampling_rate)
        )
        features[f"{axis}_spectral_entropy"] = spectral_entropy(signal, sampling_rate)
        features[f"{axis}_zero_crossings"] = zero_crossings(signal)

    return pd.DataFrame(features, index=[0])


def extract_features_from_windows(
    windows, columns_names, preselected=1, sampling_rate=100, band=(0.1, 3)
):
    """_summary_
        Extracts time-domain and frequency-domain features from a list of windows, of a specific activity
    Args:
        windows (list): a list/numpy array of windows, shape of 3d array
        columns_names (List[str]): Requires to have features named like "x-accel", "y-accel", "z-accel", "x-gyro", "y-gyro", "z-gyro". Provide a list with the ordered features names.
        sampling_rate (int): sampling rate of the sensors, default is 20
        preselected_features (bool): Uses preselected features or not, default is True
        band (tuple): (lower, upper) dont really know, default is (0.1, 3) because of chatgpt

    Returns:
        df (pd.DataFrame): of features, where each row is a window
    """
    if len(columns_names) > 6:
        print(
            "Careful, more than 6 features are not supported yet. Please use only acc and/or gyro features."
        )

    columns_names_check_accel = ["x-accel", "y-accel", "z-accel"]
    columns_names_check_gyro = ["x-gyro", "y-gyro", "z-gyro"]
    if not (
        all(x in columns_names for x in columns_names_check_accel)
        or all(x in columns_names for x in columns_names_check_gyro)
    ):
        raise ValueError("Provide a list with the respective features names.")

    combined_windows_list = []

    for idx, window in enumerate(windows):
        if idx % 10000 == 0:
            print(
                f"Extracting features from window {idx} to {idx + 10000} of {len(windows)}"
            )
        if preselected == 1:
            # print("Using preselected features (1) by article 58")
            features = preselected_features(window, columns_names, sampling_rate)
        elif preselected == 2:
            # print("Using preselected features (2) by A15")
            features = preselected_features_by_a15(
                window, columns_names, sampling_rate, bins=10
            )
        else:
            # print("Using all features")
            time_features = time_domain_features(window, columns_names, sampling_rate)
            freq_features = frequency_domain_features(
                window, columns_names, sampling_rate, band
            )
            features = pd.concat([time_features, freq_features], axis=1)

        combined_windows_list.append(features)

    combined_windows = pd.concat(
        combined_windows_list,
        axis=0,
        ignore_index=True,
    )
    print("Feature extraction completed! A dataframe of features was returned.")

    return combined_windows
