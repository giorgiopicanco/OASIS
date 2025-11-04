import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from scipy.constants import speed_of_light
from pyOASIS import gnss_freqs

# Function definitions
def geometry_free_combination_L(lambda1, lambda2, L1, L2):
    """
    Computes the geometry-free combination for carrier phase measurements.

    Parameters:
    - lambda1, lambda2: Wavelengths for the two frequencies.
    - L1, L2: Phase observations for the two frequencies.

    Returns:
    - L_GF: Result of the geometry-free combination for carrier phase.
    """
    L_GF = (lambda1*L1)-(lambda2*L2)
    # L_GF = L1 - L2
    return L_GF

def geometry_free_combination_C(P1, P2):
    """
    Computes the geometry-free combination for pseudorange code measurements.

    Parameters:
    - P1, P2: Code observations for the two frequencies.

    Returns:
    - P_GF: Result of the geometry-free combination for code.
    """
    P_GF = P2-P1
    return P_GF

# Function to fit a low-degree polynomial to data
def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)

def detect_outliers(arc_data, polynomial_fits, arc_idx, threshold_factor):
    outliers = []
    for i, (arc_values, fit) in enumerate(zip(arc_data, polynomial_fits), start=1):
        # Compute the residuals
        residuals = np.abs(arc_values - fit)
        # Compute the median of residuals
        mean_residuals = np.median(residuals)
        # Define the threshold for outlier detection
        threshold = threshold_factor * mean_residuals
        # Identify the values that are outliers
        outlier_indices = np.where(np.abs(residuals) > threshold)[0]
        # Retrieve the current arc limits
        arc_start, arc_end = arc_idx[i - 1][0], arc_idx[i - 1][-1] + 1
        # Transform arc-local outlier indices into global indices
        real_outlier_indices = outlier_indices + arc_start
        # Add outliers to the list
        outliers.extend([(i, arc_values[idx], idx, real_idx) for idx, real_idx in zip(outlier_indices, real_outlier_indices)])
    return outliers
