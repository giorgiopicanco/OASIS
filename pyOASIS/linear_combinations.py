# linear_combinations.py

# This function calculates the Melbourne-Wubbena combination
def melbourne_wubbena(f1, f2, L1, L2, P1, P2):
    phi_lw = (f1 * L1 - f2 * L2) / (f1 - f2)
    r_pn = (f1 * P1 - f2 * P2) / (f1 - f2)
    b_w = phi_lw - r_pn
    b_w = b_w / 10**7
    return b_w

