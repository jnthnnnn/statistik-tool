import tkinter as tk
from tkinter import ttk, messagebox
import scipy.stats as stats
import numpy as np

# --- Statistikfunktionen ---
def binomialtest_gui(k, n, p0, hypo_typ, alpha):
    try:
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
        output = f"""P-Wert = {result.pvalue:.5f} (α = {alpha})
Kritischer z-Wert: {crit_val:.4f}
{"Hypothese abgelehnt" if result.pvalue < alpha else "Hypothese beibehalten"}"""

        return output

    except Exception as e:
        return str(e)


def t_test_gui(x, s, n, mu0, alpha, richtung):
    try:
        x = float(x)
        s = float(s)
        n = int(n)
        mu0 = float(mu0)
        alpha = float(alpha)
        df = n - 1
        t_stat = ((x - mu0) / s) * (n ** 0.5)

        # Auswahl: Hypothesentyp bestimmen
        if richtung == "T.1.a":
            # Zweiseitig
            krit = stats.t(df).ppf(1 - alpha / 2)
            ablehnung = abs(t_stat) > krit
            richtung_text = "H1: µ ≠ µ₀\nH0: µ = µ₀"
            vergleich_text = f"|t| > {krit:.4f}"
        elif richtung == "T.1.b":
            # Einseitig rechts
            krit = stats.t(df).ppf(1 - alpha)
            ablehnung = t_stat > krit
            richtung_text = "H1: µ > µ₀\nH0: µ ≤ µ₀"
            vergleich_text = f"t > {krit:.4f}"
        elif richtung == "T.1.c":
            # Einseitig links
            krit = stats.t(df).ppf(alpha)
            ablehnung = t_stat < krit
            richtung_text = "H1: µ < µ₀\nH0: µ ≥ µ₀"
            vergleich_text = f"t < {krit:.4f}"
        else:
            raise ValueError("Unbekannter Testtyp.")

        ausgabe = f"""
Testverfahren: {richtung}
- - - - - - - - - - -
Voraussetzung erfüllt?: Metrisch oder Normalverteilt!
{richtung_text}

Testwert t = {t_stat:.4f}
Kritischer t-Wert = {krit:.4f}
Vergleich: {vergleich_text}
Signifikanzniveau α = {alpha}

Ablehnung von H0? {ablehnung}
⇒ {"H0 wird abgelehnt." if ablehnung else "H0 wird beibehalten."}
"""
        return ausgabe.strip()

    except Exception as e:
        return str(e)

def konf_gui(x, s, n, alpha):
    try:
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
Verwendeter t-Kritischer Wert: {t_val:.4f}"""
    except Exception as e:
        return str(e)
    
def b2_approximativer_binomialtest(k, n, p0, alpha, richtung):
    try:
        k = int(k)
        n = int(n)
        p0 = float(p0)
        alpha = float(alpha)

        phat = k / n
        np0 = p0 * n
        n1_p0 = (1 - p0) * n

        if np0 < 5 or n1_p0 < 5:
            return "Voraussetzungen für approximativen Binomialtest sind nicht erfüllt (np₀ oder n(1-p₀) < 5)."

        # Stetigkeitskorrektur:
        if richtung == "B.2.a":
            # Zweiseitig
            korrektur = 0.5 if k < p0 * n else -0.5
            z = (k + korrektur - p0 * n) / ((p0 * (1 - p0) * n) ** 0.5)
            krit = stats.norm.ppf(1 - alpha / 2)
            entscheidung = abs(z) > krit
            vergleich = f"|z| > {krit:.4f}"
            hyp = "H1: p ≠ p₀\nH0: p = p₀"
        elif richtung == "B.2.b":
            # Rechtseitig
            korrektur = -0.5
            z = (k + korrektur - p0 * n) / ((p0 * (1 - p0) * n) ** 0.5)
            krit = stats.norm.ppf(1 - alpha)
            entscheidung = z > krit
            vergleich = f"z > {krit:.4f}"
            hyp = "H1: p > p₀\nH0: p ≤ p₀"
        elif richtung == "B.2.c":
            # Linkseitig
            korrektur = 0.5
            z = (k + korrektur - p0 * n) / ((p0 * (1 - p0) * n) ** 0.5)
            krit = stats.norm.ppf(alpha)
            entscheidung = z < krit
            vergleich = f"z < {krit:.4f}"
            hyp = "H1: p < p₀\nH0: p ≥ p₀"
        else:
            return "Unbekannter Testtyp"

        return f"""
Testverfahren: {richtung}
- - - - - - - - - - -
{hyp}

p̂ = {phat:.4f}
Teststatistik z = {z:.4f}
Kritischer z-Wert: {krit:.4f}
Vergleich: {vergleich}
Signifikanzniveau α = {alpha}

Ablehnung von H0? {entscheidung}
⇒ {"H0 wird abgelehnt." if entscheidung else "H0 wird beibehalten."}
""".strip()

    except Exception as e:
        return str(e)

def b3_vergleich_zweier_anteile(k1, n1, k2, n2, alpha, richtung):
    try:
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
            return f"Voraussetzung nicht erfüllt: min(n1,n2) * min(p̂, 1-p̂) = {check_val:.2f} < 5"

        se = (pooled_p * (1 - pooled_p) * (1/n1 + 1/n2)) ** 0.5
        z = (p1 - p2) / se

        if richtung == "B.3.a":
            krit = stats.norm.ppf(1 - alpha / 2)
            entscheidung = abs(z) > krit
            vergleich = f"|z| > {krit:.4f}"
            hyp = "H1: p₁ ≠ p₂\nH0: p₁ = p₂"
        elif richtung == "B.3.b":
            krit = stats.norm.ppf(1 - alpha)
            entscheidung = z > krit
            vergleich = f"z > {krit:.4f}"
            hyp = "H1: p₁ > p₂\nH0: p₁ ≤ p₂"
        elif richtung == "B.3.c":
            krit = stats.norm.ppf(alpha)
            entscheidung = z < krit
            vergleich = f"z < {krit:.4f}"
            hyp = "H1: p₁ < p₂\nH0: p₁ ≥ p₂"
        else:
            return "Unbekannter Testtyp"

        return f"""
Testverfahren: {richtung}
- - - - - - - - - - -
{hyp}

p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}
Pooled p̂ = {pooled_p:.4f}
Teststatistik z = {z:.4f}
Kritischer z-Wert = {krit:.4f}
Vergleich: {vergleich}
Signifikanzniveau α = {alpha}

Ablehnung von H0? {entscheidung}
⇒ {"H0 wird abgelehnt." if entscheidung else "H0 wird beibehalten."}
""".strip()

    except Exception as e:
        return str(e)


# --- GUI ---
def show_fields(choice):
    for widget in frame_fields.winfo_children():
        widget.destroy()

    if choice == "B.1: Exakter Binomialtest":
        entries = {}
        labels = [("Anzahl Erfolge (k):", "k"),
          ("Stichprobengröße (n):", "n"),
          ("Hypothetischer Anteil p₀:", "p0"),
          ("Hypothese (zweiseitig, kleiner, größer):", "hypo"),
          ("Signifikanzniveau α (z.B. 0.05):", "alpha")]


        for text, key in labels:
            tk.Label(frame_fields, text=text).pack()
            entries[key] = tk.Entry(frame_fields)
            entries[key].pack()

        def run():
            result = binomialtest_gui(entries["k"].get(), entries["n"].get(), entries["p0"].get(),
                          entries["hypo"].get(), entries["alpha"].get())


        tk.Button(frame_fields, text="Berechnen", command=run).pack()

    elif choice == "T.1: t-Test (eine Stichprobe)":
        entries = {}
        labels = [("Mittelwert (x̄):", "x"),
                ("Standardabweichung (s):", "s"),
                ("Stichprobengröße (n):", "n"),
                ("µ₀ (hyp. Mittelwert):", "mu0"),
                ("Signifikanzniveau α (z.B. 0.05):", "alpha"),
                ("Hypothesentyp (T.1.a / T.1.b / T.1.c):", "richtung"),
                ("Rohdaten (optional, durch Kommas getrennt):", "rawdata")]

        for text, key in labels:
            tk.Label(frame_fields, text=text).pack()
            if key == "richtung":
                entries[key] = ttk.Combobox(frame_fields, state="readonly", values=["T.1.a", "T.1.b", "T.1.c"])
                entries[key].set("T.1.c")
            else:
                entries[key] = tk.Entry(frame_fields)
            entries[key].pack()

        def run():
            werte_raw = entries["rawdata"].get()
            if werte_raw:
                try:
                    zahlen = [float(w.strip()) for w in werte_raw.split(",")]
                    x = np.mean(zahlen)
                    s = np.std(zahlen, ddof=1)
                    n = len(zahlen)
                except:
                    messagebox.showerror("Fehler", "Ungültige Zahlen in Rohdaten.")
                    return
            else:
                try:
                    x = float(entries["x"].get())
                    s = float(entries["s"].get())
                    n = int(entries["n"].get())
                except:
                    messagebox.showerror("Fehler", "Bitte gültige Zahlen eingeben.")
                    return

            result = t_test_gui(
                x, s, n,
                entries["mu0"].get(),
                entries["alpha"].get(),
                entries["richtung"].get()
            )
            messagebox.showinfo("Ergebnis", result)

        # Button ganz am Ende hinzufügen
        tk.Button(frame_fields, text="Berechnen", command=run).pack(pady=10)



    elif choice == "KNF_T: Konfidenzintervall für Mittelwert":
        entries = {}
        labels = [("Mittelwert (x̄):", "x"),
                  ("Standardabweichung (s):", "s"),
                  ("Stichprobengröße (n):", "n"),
                  ("α (z.B. 0.05):", "alpha")]

        for text, key in labels:
            tk.Label(frame_fields, text=text).pack()
            entries[key] = tk.Entry(frame_fields)
            entries[key].pack()

        def run():
            result = t_test_gui(
                entries["x"].get(),
                entries["s"].get(),
                entries["n"].get(),
                entries["mu0"].get(),
                entries["alpha"].get(),
                entries["richtung"].get()
            )
            messagebox.showinfo("Ergebnis", result)

        tk.Button(frame_fields, text="Berechnen", command=run).pack()

    elif choice == "B.3: Vergleich zweier Anteile":
        entries = {}
        labels = [("Anzahl Erfolge Gruppe 1 (k1):", "k1"),
                ("Stichprobengröße Gruppe 1 (n1):", "n1"),
                ("Anzahl Erfolge Gruppe 2 (k2):", "k2"),
                ("Stichprobengröße Gruppe 2 (n2):", "n2"),
                ("Signifikanzniveau α (z.B. 0.05):", "alpha"),
                ("Hypothesentyp (B.3.a / B.3.b / B.3.c):", "richtung")]

        for text, key in labels:
            tk.Label(frame_fields, text=text).pack()
            if key == "richtung":
                entries[key] = ttk.Combobox(frame_fields, state="readonly", values=["B.3.a", "B.3.b", "B.3.c"])
                entries[key].set("B.3.a")
            else:
                entries[key] = tk.Entry(frame_fields)
            entries[key].pack()

        def run():
            result = b3_vergleich_zweier_anteile(
                entries["k1"].get(), entries["n1"].get(),
                entries["k2"].get(), entries["n2"].get(),
                entries["alpha"].get(), entries["richtung"].get()
            )
            messagebox.showinfo("Ergebnis", result)

        tk.Button(frame_fields, text="Berechnen", command=run).pack()


    elif choice == "B.2: Approximativer Binomialtest":
        entries = {}
        labels = [("Anzahl Erfolge (k):", "k"),
                ("Stichprobengröße (n):", "n"),
                ("Hypothetischer Anteil p₀:", "p0"),
                ("Signifikanzniveau α (z.B. 0.05):", "alpha"),
                ("Hypothesentyp (B.2.a / B.2.b / B.2.c):", "richtung")]

        for text, key in labels:
            tk.Label(frame_fields, text=text).pack()
            if key == "richtung":
                entries[key] = ttk.Combobox(frame_fields, state="readonly", values=["B.2.a", "B.2.b", "B.2.c"])
                entries[key].set("B.2.a")
            else:
                entries[key] = tk.Entry(frame_fields)
            entries[key].pack()

        def run():
            result = b2_approximativer_binomialtest(
                entries["k"].get(), entries["n"].get(), entries["p0"].get(),
                entries["alpha"].get(), entries["richtung"].get()
            )
            messagebox.showinfo("Ergebnis", result)

        tk.Button(frame_fields, text="Berechnen", command=run).pack()




# Hauptfenster
root = tk.Tk()
root.title("Statistik-Tool GUI")
root.geometry("400x500")

selected = tk.StringVar()

options = [
    "B.1: Exakter Binomialtest",
    "B.2: Approximativer Binomialtest",
    "B.3: Vergleich zweier Anteile",
    "T.1: t-Test (eine Stichprobe)",
    "T.2: Zwei-Stichproben t-Test",
    "KNF_T: Konfidenzintervall für Mittelwert"
]

selected.set(options[0])
tk.Label(root, textvariable=selected, font=("Arial", 12, "bold")).pack(pady=10)

dropdown = ttk.Combobox(root, textvariable=selected, values=options, state="readonly", width=35)
dropdown.pack()

frame_fields = tk.Frame(root)
frame_fields.pack(pady=10)

dropdown.bind("<<ComboboxSelected>>", lambda e: show_fields(selected.get()))
show_fields(selected.get())  # Initiale Anzeige

copyright_label = tk.Label(
    root,
    text="© J.J",
    font=("Arial", 8),
    anchor="se"
)
copyright_label.pack(side="bottom", anchor="e", padx=10, pady=5)

root.mainloop()
