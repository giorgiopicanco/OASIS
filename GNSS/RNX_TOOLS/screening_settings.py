#!/usr/bin/env python3

# screening_settings.py
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import itertools
import warnings
from numpy.polynomial import Polynomial
import gnss_freqs

def find_outliers(y_res, K):
    B = np.zeros_like(y_res)
    for i in range(len(y_res)):
        if i == 0:
            B[i] = y_res[i]
        else:
            B[i] = np.mean(y_res[:i+1])

    m = np.mean(B)
    S = np.std(B)
    threshold = K * S
    mic_out = []
    for i in range(len(y_res)):
        if i > 0:
            if abs(B[i] - m) > threshold:
                mic_out.append(i)
    return mic_out

# Função para ajustar um polinômio de baixo grau aos dados
def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)

# Função para reescalar os dados para o intervalo [-10, 10]
def rescale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    # Reescala os dados para o intervalo [0, 1]
    scaled_data = (data - min_val) / (max_val - min_val)
    # Ajusta para o intervalo [-10, 10]
    final_data = scaled_data * 20 - 10
    return final_data

def detect_outliers(arc_data, polynomial_fits, arc_idx, threshold_factor):
    outliers = []
    for i, (arc_values, fit) in enumerate(zip(arc_data, polynomial_fits), start=1):
        # Calcular os resíduos
        # residuals = arc_values - fit
        residuals = np.abs(arc_values - fit)
        # Calcular a média dos resíduos
        mean_residuals = np.median(residuals)
        # Definir o limiar para detecção de outliers
        threshold = threshold_factor * mean_residuals
        # Marcar os valores que são outliers
        outlier_indices = np.where(np.abs(residuals) > threshold)[0]
        # Recuperar os limites do arco atual
        arc_start, arc_end = arc_idx[i - 1][0], arc_idx[i - 1][-1] + 1
        # Transformar os índices dos outliers no arco nos índices dos outliers nos dados originais
        real_outlier_indices = outlier_indices + arc_start
        # Adicionar os outliers à lista
        outliers.extend([(i, arc_values[idx], idx, real_idx) for idx, real_idx in zip(outlier_indices, real_outlier_indices)])
    return outliers

def plot_selected_arcs(arc_data, arc_idx, polynomial_fits, outliers, selected_arcs):
    plt.figure(figsize=(12, 6))
    for i in selected_arcs:
        arc = arc_data[i-1]
        idx = arc_idx[i-1]
        fit = polynomial_fits[i-1]
        plt.plot(idx, arc, marker='o', linestyle='', label=f'Data {i}')
        plt.plot(idx, fit, label=f'Fit {i}', linewidth=2)

    # Adicionar outliers ao gráfico
    for arc_num, outlier_value, outlier_idx in outliers:
        arc_start, arc_end = arc_idx[arc_num - 1]
        idx_value = outlier_idx - arc_start  # Transformar o índice do outlier do arco para o índice global
        plt.scatter(arc_idx[arc_num - 1][0] + idx_value, outlier_value, color='black', marker='x', label=f'Outlier {arc_num}')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


# Definição de funções
def melbourne_wubbena_combination(f1, f2, L1, L2, P1, P2):
    """
    Melbourne-Wübbena combination calculation.

    Parameters:
    - f1, f2: Frequencies
    - L1, L2: Carrier phase measurements
    - P1, P2: Pseudo-range measurements
    - lambda_w: Wavelength of the wide-lane combination

    Returns:
    - b_w: Melbourne-Wübbena combination result
    """
    phi_lw = (f1 * L1 - f2 * L2) / (f1 - f2)
    r_pn = (f1 * P1 - f2 * P2) / (f1 - f2)  # Note: Adjusted the pseudo-range calculation
    b_w = phi_lw - r_pn
    b_w = b_w / 10**7
    # Convertendo NaN para "NaN"
    b_w[np.isnan(b_w)] = "NaN"
    return b_w


def iono_free_phase_combination(f1, f2, Phi_L1, Phi_L2):
    """
    Ionosphere-free phase combination calculation.

    Parameters:
    - f1, f2: Frequencies
    - Phi_L1, Phi_L2: Carrier phase measurements

    Returns:
    - Phi_iono_free: Ionosphere-free phase combination result
    """
    Phi_iono_free = (f1**2 * Phi_L1 - f2**2 * Phi_L2) / (f1**2 - f2**2)
    # Convertendo NaN para "NaN"
    Phi_iono_free[np.isnan(Phi_iono_free)] = "NaN"
    return Phi_iono_free / 10**7

def iono_free_range_combination(f1, f2, R_P1, R_P2):
    """
    Ionosphere-free range combination calculation.

    Parameters:
    - f1, f2: Frequencies
    - R_P1, R_P2: Pseudo-range measurements

    Returns:
    - R_iono_free: Ionosphere-free range combination result
    """
    R_iono_free = (f1**2 * R_P1 - f2**2 * R_P2) / (f1**2 - f2**2)
    # Convertendo NaN para "NaN"
    R_iono_free[np.isnan(R_iono_free)] = "NaN"
    return R_iono_free / 10**7

