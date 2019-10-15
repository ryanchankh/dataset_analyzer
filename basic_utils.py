import time
import os

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



def basic_stats_per_dim(data, dim):
    """Calcuate the basic statistics about the data along a specific dimension.
    Statistics include: mean, median, max, min
   
    Parameters:
        data (np.ndarray): data points, with dimensions (num_samples, num_features)
        dim (int): specify dimension to obtain statistics of
        
    Return:
        returned_dict (dict): a dictionary of basic stats 
    """
    vals = data[:, dim]
    return_dict = {"min": np.min(vals),
                   "max": np.max(vals),
                   "mean": np.mean(vals),
                   "median": np.median(vals)}
    return return_dict


def plot_hist_per_dim(data, dim, bins="auto"):
    """Plot the distribution of values in a specific dimension.

    Parameters:
        data (np.ndarray): data points, with dimensions (num_samples, num_features)
        dim (int): specify dimension to obtain statistics of

    Return:
        bin_vals (list): values that lie in each bin
        bin_bounds (list): bounds of each vals

    """
    y_vals = data[:, dim]

    bin_vals, bin_bounds, _ = plt.hist(y_vals, bins=bins)
    plt.title(f"Frequency of Data Points along Dim {dim}")
    plt.show()
    return bin_vals, bin_bounds
