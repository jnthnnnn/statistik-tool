import numpy as np
from scipy import stats

def binomialtest_gui(k, n, p0, hypo_typ, alpha):
    k = int(k)
    n = int(n)
    p0 = float(p0)
    alpha = float(alpha)

    if hypo_typ == "zweiseitig":
        alternative = "two-sided"
        crit_val = stats.norm.ppf(1 - alpha / 2)
    elif hypo_typ == "kleiner":
        alternative = "less"
        crit_val = stats.norm.ppf(alpha)
    elif hypo_typ == "größer":
        alternative = "greater"
        crit_val = stats.norm.ppf(1 - alpha)
    else:
        raise ValueError("Ungültige Hypothese.")

    result = stats.binomtest(k, n=n, p=p0, alternative=alternative)
    return f"""P-Wert = {result.pvalue:.5f} (α = {alpha})
Kritischer z-Wert: {crit_val:.4f}
{"Hypothese abgelehnt" if result.pvalue < alpha else "Hypothese beibehalten"}"""

def t_test_gui(x, s, n, mu0, alpha, richtung):
    x = float(x)
    s = float(s)
    n = int(n)
    mu0 = float(mu0)
    alpha = float(alpha)
    df = n - 1
    t_stat = ((x - mu0) / s) * (n ** 0.5)

    if richtung == "T.1.a":
        krit = stats.t(df).ppf(1 - alpha / 2)
        ablehnung = abs(t_stat) > krit
        vergleich_text = f"|t| > {krit:.4f}"
    elif richtung == "T.1.b":
        krit = stats.t(df).ppf(1 - alpha)
        ablehnung = t_stat > krit
        vergleich_text = f"t > {krit:.4f}"
    elif richtung == "T.1.c":
        krit = stats.t(df).ppf(alpha)
        ablehnung = t_stat < krit
        vergleich_text = f"t < {krit:.4f}"
    else:
        raise ValueError("Unbekannter Testtyp.")

    return f"""Testwert t = {t_stat:.4f}
Kritischer t-Wert = {krit:.4f}
Vergleich: {vergleich_text}
Signifikanzniveau α = {alpha}
H0 wird {'abgelehnt' if ablehnung else 'beibehalten'}."""

def konf_gui(x, s, n, alpha):
    x = float(x)
    s = float(s)
    n = int(n)
    alpha = float(alpha)

    df = n - 1
    t_val = stats.t.ppf(1 - alpha / 2, df)
    margin = t_val * (s / np.sqrt(n))
    lower = x - margin
    upper = x + margin

    return f"""{(1-alpha)*100:.1f}% Konfidenzintervall: [{lower:.4f}, {upper:.4f}]
t-Kritischer Wert: {t_val:.4f}"""

def b2_approximativer_binomialtest(k, n, p0, alpha, richtung):
    k = int(k)
    n = int(n)
    p0 = float(p0)
    alpha = float(alpha)

    phat = k / n
    np0 = p0 * n
    n1_p0 = (1 - p0) * n

    if np0 < 5 or n1_p0 < 5:
        return "Voraussetzungen für approximativen Binomialtest sind nicht erfüllt."

    if richtung == "B.2.a":
        korrektur = 0.5 if k < p0 * n else -0.5
        z = (k + korrektur - p0 * n) / ((p0 * (1 - p0) * n) ** 0.5)
        krit = stats.norm.ppf(1 - alpha / 2)
        entscheidung = abs(z) > krit
        vergleich = f"|z| > {krit:.4f}"
    elif richtung == "B.2.b":
        korrektur = -0.5
        z = (k + korrektur - p0 * n) / ((p0 * (1 - p0) * n) ** 0.5)
        krit = stats.norm.ppf(1 - alpha)
        entscheidung = z > krit
        vergleich = f"z > {krit:.4f}"
    elif richtung == "B.2.c":
        korrektur = 0.5
        z = (k + korrektur - p0 * n) / ((p0 * (1 - p0) * n) ** 0.5)
        krit = stats.norm.ppf(alpha)
        entscheidung = z < krit
        vergleich = f"z < {krit:.4f}"
    else:
        return "Unbekannter Testtyp"

    return f"""p̂ = {phat:.4f}
z = {z:.4f}
Kritischer z-Wert: {krit:.4f}
Vergleich: {vergleich}
H0 wird {'abgelehnt' if entscheidung else 'beibehalten'}."""

def b3_vergleich_zweier_anteile(k1, n1, k2, n2, alpha, richtung):
    k1 = int(k1)
    n1 = int(n1)
    k2 = int(k2)
    n2 = int(n2)
    alpha = float(alpha)

    p1 = k1 / n1
    p2 = k2 / n2
    pooled_p = (k1 + k2) / (n1 + n2)
    check_val = min(n1, n2) * min(pooled_p, 1 - pooled_p)

    if check_val < 5:
        return f"Voraussetzung nicht erfüllt (min(n1,n2)*min(p̂,1-p̂) = {check_val:.2f} < 5)"

    se = (pooled_p * (1 - pooled_p) * (1/n1 + 1/n2)) ** 0.5
    z = (p1 - p2) / se

    if richtung == "B.3.a":
        krit = stats.norm.ppf(1 - alpha / 2)
        entscheidung = abs(z) > krit
        vergleich = f"|z| > {krit:.4f}"
    elif richtung == "B.3.b":
        krit = stats.norm.ppf(1 - alpha)
        entscheidung = z > krit
        vergleich = f"z > {krit:.4f}"
    elif richtung == "B.3.c":
        krit = stats.norm.ppf(alpha)
        entscheidung = z < krit
        vergleich = f"z < {krit:.4f}"
    else:
        return "Unbekannter Testtyp"

    return f"""p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}
pooled p̂ = {pooled_p:.4f}
z = {z:.4f}
Kritischer z-Wert = {krit:.4f}
Vergleich: {vergleich}
H0 wird {'abgelehnt' if entscheidung else 'beibehalten'}."""
