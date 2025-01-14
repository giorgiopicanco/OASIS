# stec.py
# coding=utf8

import numpy as np

class tec:
    c = 299792458

    @staticmethod
    def factor(f1, f2):
        """Returns TEC factor."""
        return (1 / 40.308 *
                (f1 ** 2 * f2 ** 2) / (f1 ** 2 - f2 ** 2) * 1.0e-16)

    @staticmethod
    def get_freq(obs_code, satellite, glo_freq_num=None):
        """Return frequencies regarding to satellite system."""
        # Implement your get_freq logic here based on the original implementation

    @staticmethod
    def calculate_phase_tec(F1,F2,L1,L2,P1,P2):
        """Calculate and return phase TEC value."""


        tec_value = (tec.c / F1 * L1 -
                     tec.c / F2 * L2)

        return tec.factor(F1, F2) * tec_value


    @staticmethod
    def calculate_code_tec(F1,F2,P1,P2):
        """Calculate and return phase TEC value."""


        tec_value = (P2-P1)

        return tec.factor(F1, F2) * tec_value





