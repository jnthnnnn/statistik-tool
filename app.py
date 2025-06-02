# app.py
import streamlit as st
import numpy as np
import scipy.stats as stats
from statistik import (
    binomialtest_gui,
    t_test_gui,
    konf_gui,
    b2_approximativer_binomialtest,
    b3_vergleich_zweier_anteile
)

st.set_page_config(page_title="Statistik-Tool", layout="centered")
st.title("📊 Statistik-Tool Online")

verfahren = st.selectbox("Wähle ein Verfahren:", [
    "B.1: Exakter Binomialtest",
    "B.2: Approximativer Binomialtest",
    "B.3: Vergleich zweier Anteile",
    "T.1: t-Test (eine Stichprobe)",
    "KNF_T: Konfidenzintervall für Mittelwert"
])

if verfahren == "B.1: Exakter Binomialtest":
    k = st.number_input("Anzahl Erfolge (k):", value=0)
    n = st.number_input("Stichprobengröße (n):", value=1)
    p0 = st.number_input("Hypothetischer Anteil p₀:", value=0.5)
    hypo = st.selectbox("Hypothesentyp:", ["zweiseitig", "kleiner", "größer"])
    alpha = st.number_input("Signifikanzniveau α:", value=0.05)
    if st.button("Berechnen"):
        result = binomialtest_gui(k, n, p0, hypo, alpha)
        st.code(result)

elif verfahren == "B.2: Approximativer Binomialtest":
    k = st.number_input("Anzahl Erfolge (k):", value=0)
    n = st.number_input("Stichprobengröße (n):", value=1)
    p0 = st.number_input("Hypothetischer Anteil p₀:", value=0.5)
    alpha = st.number_input("Signifikanzniveau α:", value=0.05)
    richtung = st.selectbox("Hypothesentyp:", ["B.2.a", "B.2.b", "B.2.c"])
    if st.button("Berechnen"):
        result = b2_approximativer_binomialtest(k, n, p0, alpha, richtung)
        st.code(result)

elif verfahren == "B.3: Vergleich zweier Anteile":
    k1 = st.number_input("Erfolge Gruppe 1 (k1):", value=0)
    n1 = st.number_input("Größe Gruppe 1 (n1):", value=1)
    k2 = st.number_input("Erfolge Gruppe 2 (k2):", value=0)
    n2 = st.number_input("Größe Gruppe 2 (n2):", value=1)
    alpha = st.number_input("Signifikanzniveau α:", value=0.05)
    richtung = st.selectbox("Hypothesentyp:", ["B.3.a", "B.3.b", "B.3.c"])
    if st.button("Berechnen"):
        result = b3_vergleich_zweier_anteile(k1, n1, k2, n2, alpha, richtung)
        st.code(result)

elif verfahren == "T.1: t-Test (eine Stichprobe)":
    roh = st.text_input("Rohdaten (Komma-getrennt, optional):")
    mu0 = st.number_input("µ₀ (hyp. Mittelwert):", value=100.0)
    alpha = st.number_input("Signifikanzniveau α:", value=0.05)
    richtung = st.selectbox("Hypothesentyp:", ["T.1.a", "T.1.b", "T.1.c"])
    if roh:
        try:
            daten = [float(x.strip()) for x in roh.split(",") if x.strip()]
            x = np.mean(daten)
            s = np.std(daten, ddof=1)
            n = len(daten)
        except:
            st.error("Ungültige Rohdaten")
            x = s = n = None
    else:
        x = st.number_input("Mittelwert (x̄):", value=100.0)
        s = st.number_input("Standardabweichung (s):", value=15.0)
        n = st.number_input("Stichprobengröße (n):", value=30)

    if st.button("Berechnen") and all(v is not None for v in [x, s, n]):
        result = t_test_gui(x, s, n, mu0, alpha, richtung)
        st.code(result)

elif verfahren == "KNF_T: Konfidenzintervall für Mittelwert":
    x = st.number_input("Mittelwert (x̄):", value=100.0)
    s = st.number_input("Standardabweichung (s):", value=15.0)
    n = st.number_input("Stichprobengröße (n):", value=30)
    alpha = st.number_input("α:", value=0.05)
    if st.button("Berechnen"):
        result = konf_gui(x, s, n, alpha)
        st.code(result)

st.markdown("""
---
<p style='text-align:right;'>© J.J</p>
""", unsafe_allow_html=True)
