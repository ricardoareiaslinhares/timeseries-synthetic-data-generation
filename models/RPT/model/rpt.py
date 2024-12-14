import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.interpolate import interp1d


def augment_data(df, num_augmentations=1):
    """
    Augment the input DataFrame using rotation, permutation, and time warping.

    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data.
    num_augmentations (int): Number of augmented versions to generate.

    Returns:
    pd.DataFrame: Augmented DataFrame with original and new data.
    """

    # Convert DataFrame to numpy array
    X = df.values

    # Define augmentation functions
    def DA_Rotation(X):
        axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return np.matmul(X, axangle2mat(axis, angle))

    def DA_Permutation(X, nPerm=4, minSegLength=100):
        X_new = np.zeros(X.shape)
        idx = np.random.permutation(nPerm)
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(
            np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1)
        )
        segs[-1] = X.shape[0]
        pp = 0
        for ii in range(nPerm):
            x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1], :]
            X_new[pp : pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return X_new

    def GenerateRandomCurves(X, sigma=0.2, knot=4):
        xx = (
            np.ones((X.shape[1], 1))
            * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
        ).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        cs_x = CubicSpline(xx[:, 0], yy[:, 0])
        cs_y = CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = CubicSpline(xx[:, 2], yy[:, 2])
        return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()

    def GenerateRandomCurves_v2(X, sigma=0.2, knot=4):
        xx = np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1))
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
        x_range = np.arange(X.shape[0])

        curves = []
        for i in range(X.shape[1]):
            interpolator = interp1d(xx, yy[:, i], kind="cubic")
            curve = interpolator(x_range)
            curves.append(curve)

        return np.array(curves).T

    def GenerateRandomCurves_Gaussian(X, sigma=0.2, n_points=10):
        # Define the input space
        x_range = np.linspace(0, X.shape[0] - 1, n_points).reshape(-1, 1)

        curves = []
        for i in range(X.shape[1]):
            # Define the kernel (RBF + WhiteKernel for noise)
            kernel = 1.0 * RBF(length_scale=X.shape[0] / 4) + WhiteKernel(
                noise_level=0.1
            )

            # Create and fit the Gaussian Process model
            gp = GaussianProcessRegressor(kernel=kernel, random_state=None)
            y = np.random.normal(loc=1.0, scale=sigma, size=(n_points,))
            gp.fit(x_range, y)

            # Predict using the model
            x_pred = np.arange(X.shape[0]).reshape(-1, 1)
            y_pred, _ = gp.predict(x_pred, return_std=True)

            curves.append(y_pred)

        return np.array(curves).T

    def GenerateRandomCurves_Gaussian_2(X, sigma=0.5, n_points=10):
        # Define the input space
        x_range = np.linspace(0, X.shape[0] - 1, n_points).reshape(-1, 1)

        curves = []
        for i in range(X.shape[1]):
            # Define the kernel with a more appropriate length scale
            length_scale = (
                X.shape[0] / 2
            )  # Set length scale to half the time series length
            kernel = 1.0 * RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.1)

            # Create and fit the Gaussian Process model
            gp = GaussianProcessRegressor(
                kernel=kernel, random_state=None, n_restarts_optimizer=10
            )
            y = np.random.normal(loc=1.0, scale=sigma, size=(n_points,))
            gp.fit(x_range, y)

            # Predict using the model
            x_pred = np.arange(X.shape[0]).reshape(-1, 1)
            y_pred, _ = gp.predict(x_pred, return_std=True)

            curves.append(y_pred)

        return np.array(curves).T

    def DistortTimesteps(X, sigma=0.2):
        tt = GenerateRandomCurves_v2(X, sigma)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [
            (X.shape[0] - 1) / tt_cum[-1, 0],
            (X.shape[0] - 1) / tt_cum[-1, 1],
            (X.shape[0] - 1) / tt_cum[-1, 2],
        ]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
        tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
        return tt_cum

    def DA_TimeWarp(X, sigma=0.2):
        tt_new = DistortTimesteps(X, sigma)
        X_new = np.zeros(X.shape)
        x_range = np.arange(X.shape[0])
        X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
        X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
        X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
        return X_new

    # Generate augmented data
    augmented_data = []
    for _ in range(num_augmentations):
        X_aug = X.copy()
        X_aug = DA_Rotation(X_aug)
        X_aug = DA_Permutation(X_aug)
        X_aug = DA_TimeWarp(X_aug)
        augmented_data.append(X_aug)

    # Combine all augmented data
    combined_data = np.vstack(augmented_data)

    # Create a new DataFrame with the combined data
    columns = df.columns
    index = pd.RangeIndex(start=0, stop=len(combined_data), step=1)
    augmented_df = pd.DataFrame(combined_data, columns=columns, index=index)

    return augmented_df


# Example usage:
# Assuming 'df' is your input DataFrame
# augmented_df = augment_data(df, num_augmentations=3)
