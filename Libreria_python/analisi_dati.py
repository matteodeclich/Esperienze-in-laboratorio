import numpy as np
from scipy import stats
from uncertainties import ufloat
import uncertainties.unumpy as unp
 
 
def minquadw_kx(x, y, sigma_y, nome="k"):
    """
    Minimi quadrati pesati: y = kx.
    Restituisce k come oggetto ufloat.
 
    Parametri
    ---------
    nome : str
        Nome della variabile usato nella stampa (default "k")
    """
    x, y, sigma_y = np.array(x), np.array(y), np.array(sigma_y)
    w = 1.0 / (sigma_y**2)
 
    sum_w_x2 = np.sum(w * x**2)
    sum_w_xy = np.sum(w * x * y)
 
    k_val   = sum_w_xy / sum_w_x2
    sigma_k = np.sqrt(1.0 / sum_w_x2)
 
    k = ufloat(k_val, sigma_k)
 
    print(f"Il risultato del fit y = {nome}·x è:")
    print(f"  {nome} = {k:.8P}")
 
    return k
 
 
def minquadw_abx(x, y, sigma_y, nome_a="a", nome_b="b"):
    """
    Minimi quadrati pesati: y = a + bx.
    Restituisce a, b come oggetti ufloat e cov_ab come float.
 
    Parametri
    ---------
    nome_a : str
        Nome dell'intercetta usato nella stampa (default "a")
    nome_b : str
        Nome del coefficiente angolare usato nella stampa (default "b")
    """
    x, y, sigma_y = np.array(x), np.array(y), np.array(sigma_y)
    w = 1.0 / (sigma_y**2)
 
    sum_w   = np.sum(w)
    sum_wx  = np.sum(w * x)
    sum_wy  = np.sum(w * y)
    sum_wx2 = np.sum(w * x**2)
    sum_wxy = np.sum(w * x * y)
 
    Delta = sum_w * sum_wx2 - (sum_wx)**2
 
    a_val = (sum_wx2 * sum_wy  - sum_wx  * sum_wxy) / Delta
    b_val = (sum_w   * sum_wxy - sum_wx  * sum_wy)  / Delta
 
    sigma_a = np.sqrt(sum_wx2 / Delta)
    sigma_b = np.sqrt(sum_w   / Delta)
    cov_ab  = -sum_wx / Delta
 
    a = ufloat(a_val, sigma_a)
    b = ufloat(b_val, sigma_b)
 
    print(f"Il risultato del fit y = {nome_a} + {nome_b}·x è:")
    print(f"  {nome_a} = {a:.8P}")
    print(f"  {nome_b} = {b:.8P}")
    print(f"  Covarianza cov({nome_a}, {nome_b}) = {cov_ab:.8g}")
 
    return a, b, cov_ab
 
 
def chi_quadro(y_misurati, y_attesi, sigma_y, n_parametri=2):
    """
    Calcola il Chi Quadro.
    Accetta y_misurati e y_attesi sia come liste di numeri puri,
    sia come array di oggetti ufloat.
    """
 
    def get_val(item):
        return item.nominal_value if hasattr(item, "nominal_value") else item
 
    y_m = np.array([get_val(m) for m in y_misurati])
    y_a = np.array([get_val(a) for a in y_attesi])
    s_y = np.array([get_val(s) for s in sigma_y])
 
    chi2 = np.sum(((y_m - y_a) / s_y)**2)
    ndof = len(y_m) - n_parametri
 
    if ndof > 0:
        p_value      = stats.chi2.sf(chi2, ndof)
        chi2_ridotto = chi2 / ndof
    else:
        p_value      = np.nan
        chi2_ridotto = np.nan
 
    print(f"Il risultato del test del χ² è:")
    print(f"  χ²               = {chi2:.8g}")
    print(f"  Gradi di libertà = {ndof}")
    print(f"  χ² ridotto       = {chi2_ridotto:.8g}")
    print(f"  Il p-value associato è: {p_value:.8g}")
 
    return chi2, ndof, chi2_ridotto, p_value
 
 
def t_test(misura1, misura2, nome1="misura 1", nome2="misura 2"):
    """
    Test di compatibilità tra due misure: accetta due oggetti ufloat.
    Restituisce il valore |t| (float).
 
    Parametri
    ---------
    nome1, nome2 : str
        Nomi delle misure usati nella stampa
    """
    diff = misura1 - misura2
    t = abs(diff.nominal_value) / diff.std_dev
 
    print(f"Test di compatibilità tra {nome1} e {nome2}: |t| = {t:.8g}")
 
    return t
 
 
def compatibilita_valore(misura, valore_vero, nome="misura"):
    """
    Test di compatibilità tra una misura (ufloat) e un valore di riferimento (float).
    Restituisce il valore |t| (float).
 
    Parametri
    ---------
    nome : str
        Nome della misura usato nella stampa
    """
    diff = np.abs(misura.nominal_value - valore_vero)
    t = diff / misura.std_dev
 
    print(f"Compatibilità di {nome} con {valore_vero:.8g}: |t| = {t:.8g}")
 
    return t

