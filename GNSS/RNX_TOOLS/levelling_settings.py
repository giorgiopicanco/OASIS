import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time
from scipy.constants import speed_of_light
import gnss_freqs

# Definição de funções
def geometry_free_combination_L(lambda1, lambda2, L1, L2):
    """
    Calcula a combinação livre de geometria para a fase.

    Parâmetros:
    - lambda1, lambda2: Comprimentos de onda para as duas frequências.
    - L1, L2: Observações de fase para as duas frequências.

    Retorna:
    - L_GF: Resultado da combinação livre de geometria para a fase.
    """
    L_GF = (lambda1*L1)-(lambda2*L2)
    # L_GF = L1 - L2
    return L_GF



# Definição de funções
def geometry_free_combination_C(P1, P2):
    """
    Calcula a combinação livre de geometria para o código.

    Parâmetros:
    - P1, P2: Observações de código para as duas frequências.

    Retorna:
    - P_GF: Resultado da combinação livre de geometria para o código.
    """
    P_GF = P2-P1
    return P_GF

    # gf = gf / 10**7
    # Convertendo NaN para "NaN"
    # gf2[np.isnan(gf2)] = "NaN"
    return gf2


# Função para ajustar um polinômio de baixo grau aos dados
def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.polyval(coeffs, x)



def detect_outliers(arc_data, polynomial_fits, arc_idx, threshold_factor):
    outliers = []
    for i, (arc_values, fit) in enumerate(zip(arc_data, polynomial_fits), start=1):
        # Calcular os resíduos

        residuals = np.abs(arc_values - fit)
        # Calcular a média dos resíduos
        mean_residuals = np.median(residuals)
        # print(mean_residuals)
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
